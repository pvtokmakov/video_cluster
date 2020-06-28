"""
General agent class for training IR and LA models.
"""
import os
import sys
import copy
import json
import logging
import numpy as np
from tqdm import tqdm
from itertools import product, chain
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.backends.cudnn as cudnn

from src.utils.utils import \
    save_checkpoint as save_snapshot, \
    copy_checkpoint as copy_snapshot, \
    AverageMeter, adjust_learning_rate, exclude_bn_weight_bias_from_weight_decay
from src.utils.setup import print_cuda_statistics
from src.datasets.kinetics import Kinetics
from src.objectives.localagg import LocalAggregationLossModule, MemoryBank, Kmeans
from src.objectives.instance import InstanceDiscriminationLossModule

from src.datasets.spatial_transforms import MultiScaleRandomCrop, Compose, RandomHorizontalFlip, RandomGrayscale, ColorJitter, ToTensor
from src.datasets.temporal_transforms import TemporalRandomCrop, TemporalCenterCrop
from src.datasets.target_transforms import ClassLabel

from src.models import resnet3d


class BaseAgent(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

        self._set_seed()  # set seed as early as possible

        self._load_datasets()
        self.train_loader, self.train_len = self._create_dataloader(self.train_dataset, shuffle=True)
        self.val_loader, self.val_len = self._create_dataloader(self.val_dataset, shuffle=False)

        self._choose_device()
        self._create_model()
        self._create_optimizer()

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_val_iteration = 0

        # we need these to decide best loss
        self.current_loss = 0
        self.current_val_metric = 0
        self.best_val_metric = 0
        self.iter_with_no_improv = 0

        print(self.config.summary_dir)

        try:  # hack to handle different versions of TensorboardX
            self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir + "/" + config.exp_name)
        except:
            self.summary_writer = SummaryWriter(logdir=self.config.summary_dir + "/" + config.exp_name)

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _choose_device(self):
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = self.config.seed
        if self.cuda: torch.cuda.manual_seed(self.manual_seed)

        if self.cuda:
            if not isinstance(self.config.gpu_device, list):
                self.config.gpu_device = [self.config.gpu_device]
            num_gpus = len(self.config.gpu_device)
            self.multigpu = num_gpus > 1 and torch.cuda.device_count() > 1

            if not self.multigpu:  # e.g. just 1 GPU
                gpu_device = self.config.gpu_device[0]
                self.logger.info("User specified 1 GPU: {}".format(gpu_device))
                self.device = torch.device("cuda")
                torch.cuda.set_device(gpu_device)
            else:
                gpu_devices = ','.join([str(_gpu_id) for _gpu_id in self.config.gpu_device])
                self.logger.info("User specified {} GPUs: {}".format(
                    num_gpus, gpu_devices))
                self.device = torch.device("cuda")

            self.gpu_devices = self.config.gpu_device
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

    def _load_datasets(self):
        raise NotImplementedError

    def _create_dataloader(self, dataset, shuffle=True):
        dataset_size = len(dataset)
        loader = DataLoader(dataset, batch_size=self.config.optim_params.batch_size,
                            shuffle=shuffle, pin_memory=True,
                            num_workers=self.config.data_loader_workers)

        return loader, dataset_size

    def _create_model(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
            self.cleanup()
        except KeyboardInterrupt as e:
            self.logger.info("Interrupt detected. Saving data...")
            self.backup()
            self.cleanup()
            raise e

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.config.num_epochs):
            adjust_learning_rate(epoch=epoch, opt_params=self.config.optim_params, optimizer=self.optim)
            self.current_epoch = epoch
            self.train_one_epoch()
            if (self.config.validate and
                    epoch % self.config.optim_params.validate_freq == 0):
                self.validate()  # validate every now and then
            self.save_checkpoint()

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def backup(self):
        """
        Backs up the model upon interrupt
        """
        self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir, "all_scalars.json".format()))
        self.summary_writer.close()

        self.logger.info("Backing up current version of model...")
        self.save_checkpoint(filename='backup.pth.tar')

    def finalise(self):
        """
        Do appropriate saving after model is finished training
        """
        self.summary_writer.export_scalars_to_json(os.path.join(self.config.summary_dir, "all_scalars.json".format()))
        self.summary_writer.close()

        self.logger.info("Saving final versions of model...")
        self.save_checkpoint(filename='final.pth.tar')

    def cleanup(self):
        """
        Undo any global changes that the Agent may have made
        """
        if self.multigpu:
            del os.environ['CUDA_VISIBLE_DEVICES']



class KineticsAgent(BaseAgent):
    def __init__(self, config):
        super(KineticsAgent, self).__init__(config)

        self._init_memory_bank()
        self._init_cluster_labels()

        loss_class = globals()[self.config.loss_params.loss]  # toggle between ID and LA
        self._init_loss_function(loss_class, self.memory_bank, self.cluster_labels)

        self.km = None  # will be populated by a kmeans model
        self.parallel_helper_idxs = torch.arange(len(self.gpu_devices)).to(self.device)

        # if user did not specify kmeans_freq, then set to constant
        if self.config.loss_params.kmeans_freq is None:
            self.config.loss_params.kmeans_freq = (
                    len(self.train_dataset) //
                    self.config.optim_params.batch_size)

        self.val_acc = []
        self.train_loss = []
        self.train_extra = []

        cudnn.benchmark = True

        self.first_iteration_kmeans = True

    def _init_memory_bank(self, attr_name='memory_bank'):
        data_len = len(self.train_dataset)
        memory_bank = MemoryBank(data_len, self.config.model_params.out_dim, self.gpu_devices)
        setattr(self, attr_name, memory_bank)

    def load_memory_bank(self, memory_bank):
        self._load_memory_bank(memory_bank, attr_name='memory_bank')

    def _load_memory_bank(self, memory_bank, attr_name='memory_bank'):
        _bank = memory_bank.bank_broadcast[0].cpu()
        _bank_broadcast = torch.cuda.comm.broadcast(_bank, self.gpu_devices)
        self._get_memory_bank(attr_name).bank_broadcast = _bank_broadcast
        self._get_loss_func('loss_fn').module.memory_bank_broadcast = self._get_memory_bank(attr_name).bank_broadcast

    def _load_cluster_labels(self, cluster_labels, attr_name='cluster_labels'):
        if type(cluster_labels) is list:
            cluster_labels = cluster_labels[0]
        
        broadcast_cluster_labels = torch.cuda.comm.broadcast(cluster_labels, self.gpu_devices)

        setattr(self, attr_name, broadcast_cluster_labels)
        self._get_loss_func('loss_fn').module.cluster_label_broadcast = getattr(self, attr_name)

    def get_memory_bank(self):
        return self._get_memory_bank(attr_name='memory_bank')

    def _get_memory_bank(self, attr_name='memory_bank'):
        return getattr(self, attr_name)

    def _get_loss_func(self, attr_name='loss_fn'):
        return getattr(self, attr_name)

    def _init_cluster_labels(self, attr_name='cluster_labels'):
        no_kmeans_k = self.config.loss_params.n_kmeans  # how many wil be train
        data_len = len(self.train_dataset)
        # initialize cluster labels
        cluster_labels = torch.arange(data_len).long()
        cluster_labels = cluster_labels.unsqueeze(0).repeat(no_kmeans_k, 1)
        broadcast_cluster_labels = torch.cuda.comm.broadcast(cluster_labels, self.gpu_devices)
        setattr(self, attr_name, broadcast_cluster_labels)

    def _init_loss_function(self, loss_class, memory_bank, cluster_label=None, attr_name='loss_fn'):
        loss_fn = loss_class(memory_bank.bank_broadcast,
                             cluster_label,
                             k=self.config.loss_params.k,
                             t=self.config.loss_params.t,
                             m=self.config.loss_params.m)
        loss_fn = nn.DataParallel(loss_fn, device_ids=self.gpu_devices).to(self.device)
        setattr(self, attr_name, loss_fn)

    def _load_image_transforms(self):

        scales = [1.0]
        for i in range(1, 5):
            scales.append(scales[-1] * 0.84081289641525)

        image_size = self.config.data_params.image_size
            
        train_spatial_transforms = Compose([
            MultiScaleRandomCrop(scales, image_size),
            RandomGrayscale(p=0.2),
            ColorJitter(0.4, 0.4, 0.4, 0.4),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[114.7748 / 255, 107.7354 / 255, 99.4750 / 255],
                                 std=[1, 1, 1]),
        ])

        test_spatial_transforms = Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[114.7748 / 255, 107.7354 / 255, 99.4750 / 255],
                                 std=[1, 1, 1]),
        ])

        train_temporal_transform = transforms.Compose([TemporalRandomCrop(self.config.data_params.sample_duration)])
        test_temporal_transform = transforms.Compose([TemporalRandomCrop(self.config.data_params.sample_duration)])

        train_target_transform = ClassLabel()
        test_target_transform = ClassLabel()

        return train_spatial_transforms, train_temporal_transform, train_target_transform, test_spatial_transforms, \
               test_temporal_transform, test_target_transform

    def _load_datasets(self):
        train_spatial_transforms, train_temporal_transform, train_target_transform, test_spatial_transforms, \
        test_temporal_transform, test_target_transform = self._load_image_transforms()
        # build training dataset
        train_dataset = Kinetics(root_path=self.config.data_params.dataset_path, annotation_path=self.config.data_params.annotation_path, 
                                 train=True, spatial_transform=train_spatial_transforms,
                                 temporal_transform=train_temporal_transform, target_transform=train_target_transform,
                                 load_fvs=self.config.data_params.load_fvs, fv_path=self.config.data_params.fv_path)
        # build validation set
        val_dataset = Kinetics(root_path=self.config.data_params.dataset_path, annotation_path=self.config.data_params.annotation_path, 
                                 train=False, spatial_transform=test_spatial_transforms,
                                 temporal_transform=test_temporal_transform, target_transform=test_target_transform)

        # save some stuff to config
        self.config.data_params.n_channels = 3

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        train_samples = train_dataset.data
        train_labels = [train_samples[i]['label'] for i in range(len(train_samples))]
        self.train_ordered_labels = np.array(train_labels)

    def _create_model(self):
        if self.config.model_params.resnet_version == "resnet18":
            model = resnet3d.resnet18(
                num_classes=self.config.model_params.out_dim,
                shortcut_type="B",
                sample_size=self.config.data_params.image_size,
                sample_duration=self.config.data_params.sample_duration)
        elif self.config.model_params.resnet_version == "resnet34":
            model = resnet3d.resnet34(
                num_classes=self.config.model_params.out_dim,
                shortcut_type="B",
                sample_size=self.config.data_params.image_size,
                sample_duration=self.config.data_params.sample_duration)
        else:
            raise NotImplementedError

        model = nn.DataParallel(model)
        model = model.to(self.device)
        cudnn.benchmark = True

        self.model = model

    def _set_models_to_eval(self):
        self.model = self.model.eval()

    def _set_models_to_train(self):
        self.model = self.model.train()

    def _create_optimizer(self):
        # Exclude batch norm weights and bias from weight decay
        parameters = exclude_bn_weight_bias_from_weight_decay(self.model,
                                                              weight_decay=self.config.optim_params.weight_decay)

        self.optim = torch.optim.SGD(parameters,
                                     lr=self.config.optim_params.learning_rate,
                                     momentum=self.config.optim_params.momentum,
                                     weight_decay=self.config.optim_params.weight_decay)

    def train_one_epoch(self):
        num_batches = self.train_len // self.config.optim_params.batch_size
        tqdm_batch = tqdm(total=num_batches,
                          desc="[Epoch {}, lr {}]".format(self.current_epoch, self.optim.param_groups[0]['lr']))

        self._set_models_to_train()  # turn on train mode
        epoch_loss = AverageMeter()

        for batch_i, (indices, images, labels) in enumerate(self.train_loader):
            batch_size = images.size(0)

            # cast elements to CUDA
            indices = indices.to(self.device, non_blocking=True)
            images = images.to(self.device, non_blocking=True)

            # do a forward pass
            outputs = self.model(images)

            # compute the loss
            loss, new_data_memory = self.loss_fn(indices, outputs, self.parallel_helper_idxs)
            loss = torch.mean(loss)  # average the loss from multi-gpus to obtain a scalar value

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            with torch.no_grad():
                self.memory_bank.update(indices, new_data_memory)
                self.loss_fn.module.memory_bank_broadcast = self.memory_bank.bank_broadcast  # update loss_fn

                if (self.config.loss_params.loss == 'LocalAggregationLossModule' and not self.config.cluster_checkpoint_dir and
                        (self.first_iteration_kmeans or batch_i == self.config.loss_params.kmeans_freq)):

                    if self.first_iteration_kmeans:
                        self.first_iteration_kmeans = False

                    self.logger.info('Fitting K-means with FAISS')

                    # get kmeans clustering (update our saved clustering)
                    k = [self.config.loss_params.kmeans_k for _ in
                         range(self.config.loss_params.n_kmeans)]

                    # NOTE: we use a different gpu for FAISS otherwise cannot fit onto memory
                    self.km = Kmeans(k, self.memory_bank, [0, 1], fvs=self.train_dataset.fvs)
                    cluster_labels = self.km.compute_clusters()

                    # broadcast the cluster_labels after modification
                    for i in range(len(self.gpu_devices)):
                        device = self.loss_fn.module.cluster_label_broadcast[i].device
                        self.loss_fn.module.cluster_label_broadcast[i] = cluster_labels.to(device)

            epoch_loss.update(loss.item(), batch_size)
            tqdm_batch.set_postfix({"Loss": epoch_loss.avg})

            self.summary_writer.add_scalars("epoch/loss", {'loss': epoch_loss.val},
                                            self.current_iteration)

            self.train_loss.append(epoch_loss.val)

            self.current_iteration += 1
            tqdm_batch.update()

        self.current_loss = epoch_loss.avg
        tqdm_batch.close()

    def validate(self):
        # For validation, for each image, we find the closest neighbor in the
        # memory bank (training data), take its class! We compute the accuracy.

        num_batches = self.val_len // self.config.optim_params.batch_size
        tqdm_batch = tqdm(total=num_batches, desc="[Val]")

        self._set_models_to_eval()
        num_correct = 0.
        num_total = 0.

        with torch.no_grad():
            for _, images, labels in self.val_loader:
                batch_size = images.size(0)

                # cast elements to CUDA
                images = images.to(self.device)
                outputs = self.model(images)

                # use memory bank to ge the top 1 neighbor
                # from the training dataset
                all_dps = self.memory_bank.get_all_dot_products(outputs)
                _, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
                neighbor_idxs = neighbor_idxs.squeeze(1)  # shape: batch_size
                neighbor_idxs = neighbor_idxs.cpu().numpy()  # convert to numpy
                # fetch the actual label of each example
                neighbor_labels = self.train_ordered_labels[neighbor_idxs]
                neighbor_labels = torch.from_numpy(neighbor_labels).long()

                num_correct += torch.sum(neighbor_labels == labels).item()
                num_total += batch_size

                tqdm_batch.set_postfix({"Val Accuracy": num_correct / num_total})
                tqdm_batch.update()

        self.summary_writer.add_scalars("Val/accuracy",
                                        {'accuracy': num_correct / num_total},
                                        self.current_epoch)

        self.current_val_iteration += 1
        self.current_val_metric = num_correct / num_total

        if self.current_val_metric >= self.best_val_metric:  # NOTE: >= for accuracy
            self.best_val_metric = self.current_val_metric

        tqdm_batch.close()

        # store the validation metric from every epoch
        self.val_acc.append(self.current_val_metric)

        return self.current_val_metric

    def load_checkpoint(self, filename, checkpoint_dir=None, cluster_label_dir=None,
                        load_memory_bank=True, load_model=True,
                        load_optim=True, load_epoch=True):
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        checkpoint_filename = os.path.join(checkpoint_dir, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(checkpoint_filename))
            checkpoint = torch.load(checkpoint_filename, map_location='cpu')

            if load_epoch:
                self.current_epoch = checkpoint['epoch'] + 1
                self.current_iteration = checkpoint['iteration']
                self.current_val_iteration = checkpoint['val_iteration']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(model_state_dict)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

                lr_pretrained = self.optim.param_groups[0]['lr']
                lr_config = self.config.optim_params.learning_rate

                # Change learning rate
                if not lr_pretrained == lr_config:
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = self.config.optim_params.learning_rate

            if load_memory_bank:  # load memory_bank
                self._load_memory_bank(checkpoint['memory_bank'])

            if cluster_label_dir is not None:
                cluster_checkpoint = torch.load(os.path.join(cluster_label_dir, filename), map_location='cpu')
                self._load_cluster_labels(cluster_checkpoint['cluster_labels'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {}) "
                             "with val acc = {}\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration'], checkpoint['val_acc']))
        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e

    def copy_checkpoint(self, filename="checkpoint.pth.tar"):
        if self.current_epoch % self.config.copy_checkpoint_freq == 0:
            copy_snapshot(
                filename=filename, folder=self.config.checkpoint_dir,
                copyname='checkpoint_epoch{}.pth.tar'.format(self.current_epoch),
            )

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'memory_bank': self.memory_bank,
            'cluster_labels': self.cluster_labels,
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'loss': self.current_loss,
            'val_iteration': self.current_val_iteration,
            'val_metric': self.current_val_metric,
            'config': self.config,
            'val_acc': np.array(self.val_acc),
            'train_loss': np.array(self.train_loss),
            'train_extra': np.array(self.train_extra),
        }

        # if we aren't validating, then every time we save is the
        # best new epoch!
        is_best = ((self.current_val_metric == self.best_val_metric) or
                   not self.config.validate)
        save_snapshot(out_dict, is_best, filename=filename,
                      folder=self.config.checkpoint_dir)

