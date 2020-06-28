import torch
from torch.autograd import Variable
import time
import sys
import numpy as np

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger, writer):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies1 = AverageMeter()
    accuracies5 = AverageMeter()

    end_time = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            acc1 = 0
            data_time.update(time.time() - end_time)

            if not opt.no_cuda:
                targets = targets.cuda(async=True)
            if opt.dataset == "hmdb51":
                targets -= 1

            outputs = model(inputs.cuda())
            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0))

            acc1 = calculate_accuracy(outputs, targets)
            acc5 = calculate_accuracy(outputs, targets, 5)
            accuracies5.update(acc5, inputs.size(0))
            accuracies1.update(acc1, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies1))

    dataset = "ucf"
    if opt.dataset == "hmdb51":
        dataset = "hmdb"

    writer.add_scalar('%s/val_top5' % dataset, accuracies5.avg, epoch)
    writer.add_scalar('%s/val_top1' % dataset, accuracies1.avg, epoch)
    writer.add_scalar('%s/val_loss' % dataset, losses.avg, epoch)

    return losses.avg


def val_final(data_loader, model, opt):
    print('Final validation')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies1 = AverageMeter()
    accuracies5 = AverageMeter()

    end_time = time.time()
    with torch.no_grad():
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            inputs1, inputs2, inputs3, inputs4, inputs5, targets = inputs

            if not opt.no_cuda:
                targets = targets.cuda(async=True)
            if opt.dataset == "hmdb51":
                targets -= 1

            outputs1 = model(inputs1)
            outputs2 = model(inputs2)
            outputs3 = model(inputs3)
            outputs4 = model(inputs4)
            outputs5 = model(inputs5)

            outputs = (outputs1 + outputs2 + outputs3 + outputs4 + outputs5) / 5.0

            acc1 = calculate_accuracy(outputs, targets)
            acc5 = calculate_accuracy(outputs, targets, 5)

            accuracies5.update(acc5, inputs1.size(0))
            accuracies1.update(acc1, inputs1.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                acc=accuracies1))
