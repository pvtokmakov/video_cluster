# Unsupervised Learning of Video Representations via Dense Trajectory Clustering

This is an implementation of the [Unsupervised Learning of Video Representations via Dense Trajectory Clustering](https://arxiv.org/abs/2006.15731) algorithm.

The codebased is built upon [Local Aggregation](https://github.com/neuroailab/LocalAggregation-Pytorch) and [3D ResNet](https://github.com/kenshohara/3D-ResNets-PyTorch).  

## Prerequisites

* Linux
* Pytorch 1.2.0
* [Faiss](https://github.com/facebookresearch/faiss)
* tqdm
* dotmap
* tensorboardX
* sklearn
* pandas

## Unsupervised representation learning

### Dataset preprocessing
Training is done on the [Kinetics-400 dataset](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Download it and preprocess as follows.
```
cd 3D-ResNet
```

* Convert from avi to jpg files using ```utils/video_jpg_kinetics.py```

```bash
python utils/video_jpg_kinetics.py AVI_VIDEO_DIRECTORY JPG_VIDEO_DIRECTORY
```

* Generate n_frames files using ```utils/n_frames_kinetics.py```

```bash
python utils/n_frames_kinetics.py JPG_VIDEO_DIRECTORY
```

* Generate annotation file in json format using ```utils/kinetics_json.py```
  * The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.

```bash
python utils/kinetics_json.py TRAIN_CSV_PATH VAL_CSV_PATH TEST_CSV_PATH DST_JSON_APTH
```

If you want to use our precomuted IDT clusters for training, please use the Kinetics annotation file provided in this codebase (splits.json). If you find that some videos are missing in you local copy of Kinetics, then you'll have to recompute the clusters using the cluster_fv.py script below, otherwise the correspondence between cluster labels and videos will be broken.

### Runtime Setup
```
cd LocalAggregation
source init_env.sh
```

### Pretrained models
We provide several models trained using our Video LA + IDT prior objective, as well as precomputed clusters for the training set of Kinetics-400, under this [link](https://drive.google.com/file/d/1i3Vn_85Fo94BINHgpMaLNvZOKPfS3lvf/view?usp=sharing) (for the varaints trained on 370k videos we skipped the last tuning stage due to memory issues). In addition, this archive contains models finetuned on UCF101 and HMDB51, which are reported in the state-of-the-art comparison section of the paper.  

### Training using precomputed IDT descriptors
Begin with training a 3D ResNet with an IR objective for 40 epochs. This is done as a warmup step. Don't forget to update data and experiment paths in the config file.
```
CUDA_VISIBLE_DEVICES=0 python scripts/instance.py ./config/kinetics_ir.json 
```
Then specify `instance_exp_dir` in `./config/kinetics_la.json` to point to the IR model you've just trained, and run the following command to trasfer IDT representations to the 3D ResNet via non-parametric clustering:
```
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/localagg.py ./config/kinetics_la.json
```
To run the final fine-tuning stage, specify `instance_exp_dir` in `./config/kinetics_la_tune.json` to point to the model trained with IDTs, and run the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2 python scripts/localagg.py ./config/kinetics_la_tune.json
```

### Recomputing and clustering IDT descriptors
We provide precomputed Fisher vector-encoded IDT descriptors for the Kinetics dataset under this [link](https://drive.google.com/file/d/1I5ZWlYJfFxXhPrv6gRq1jZJah85usd1H/view?usp=sharing).

If you wish to recompute them, you will need to first download and install the original [IDT implementation](https://lear.inrialpes.fr/people/wang/improved_trajectories).
This codes takes person detections as input. You can download the detections we used [here](https://drive.google.com/file/d/1CDX8qkhsx9ygL27VG8UQpzAipa3MeHPu/view?usp=sharing).

Next, estimate the model (PCA, GMM) parameters used in Fisher vector encoding. To this end, first sample 3500 videos from Kinetics at random, and compute IDTs for them, using the script bellow (don't forget to update paths to the IDT implementation).
```
sh src/idt/run_idt.sh PATH_TO_VIDEO PATH_TO_BOXES OUTPUT_NAME PATH_TO_IDTS
``` 
Then run the following script to estimate model parameters based on the computed IDTs. The parameters will be saved to the same directory as the IDTs.
```
python src/idt/compute_fv_models.py --idt_path PATH_TO_IDTS
```

Now you can compute the Fisher vector encoded IDT descriptors for training set of Kinetics. The following script takes a category as input, so the in can be parallelized 400-way on a CPU cluster (pleas update the path to a temporary folder insight the script, which is used to store raw IDTs).
```
python src/idt/extract_idt.py --category CATEGORY_NAME --model_path PATH_TO_IDTS --videos_path PATH_TO_TRAIN_VIDEOS --boxes_path PATH_TO_BOXES --out_path FV_OUTPUT_PATH
```

Finally, to cluster descriptors, run the following script.
```
python src/idt/cluster_fv.py --k 6000 --num_c 3 --frames_path PATH_TO_FRAMES --annotation_path PATH_TO_ANNOTATIONS_JSON --fv_path FV_OUTPUT_PATH --clusters_path PATH_TO_OUTPUT_CLUSTERS_DIRECTORY --processed_annotation_path PATH_TO_OUTPUT_ANNOTATIONS_JSON --gpu 0 1
```
This script produces a clustering assignement for the training set videos, and a new annotation file. Make sure to use this file in all the config files to ensure correct correspondence between videos and cluster labels.

## Transfer learning
```
cd 3D-ResNet
```

### Dataset preprocessing
Download and pre-process [UCF101](http://crcv.ucf.edu/data/UCF101.php) and [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) datasets as follows.

```bash
python utils/video_jpg_ucf101_hmdb51.py AVI_VIDEO_DIRECTORY JPG_VIDEO_DIRECTORY
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py JPG_VIDEO_DIRECTORY
```

* Generate annotation file in json format using ```utils/ucf101_json.py``` and ```utils/hmdb51_json.py```

```bash
python utils/ucf101_json.py ANNOTATION_DIR_PATH
python utils/hmdb51_json.py ANNOTATION_DIR_PATH
```

### Finetuning
On UCF101:
```bash
python main.py --video_path PATH_TO_FRAMES --annotation_path PATH_TO_ANNOTATION --result_path OUTPUT_MODEL_PATH --dataset ucf101 --n_finetune_classes 101 --model resnet --model_depth 18 --resnet_shortcut B --batch_size 128 --n_threads 16 --gpu 0 --pretrain_path PATH_TO_PRETRAINED_MODEL  --checkpoint 10 --ft_begin_index 2 --n_epochs 40 --lr_patience 5  --n_scales 2 --train_crop random
```

On HMDB51:
```bash
python main.py --video_path PATH_TO_FRAMES --annotation_path PATH_TO_ANNOTATION --result_path OUTPUT_MODEL_PATH --dataset hmdb51 --n_finetune_classes 101 --model resnet --model_depth 18 --resnet_shortcut B --batch_size 128 --n_threads 16 --gpu 0 --pretrain_path PATH_TO_PRETRAINED_MODEL  --checkpoint 10 --ft_begin_index 3 --n_epochs 30 --lr_patience 5  --n_scales 2 --train_crop random
```

### Evaluation
On UCF101:
```bash
python main.py --video_path PATH_TO_FRAMES --annotation_path PATH_TO_ANNOTATION --dataset ucf101 --n_classes 101 --model resnet --model_depth 18 --resnet_shortcut B --batch_size 128 --n_threads 16 --gpu 0 --test --no_train --no_val --resume_path OUTPUT_MODEL_PATH/save_40.pth
```

On HMDB51:
```bash
python main.py --video_path PATH_TO_FRAMES --annotation_path PATH_TO_ANNOTATION --dataset hmdb51 --n_classes 101 --model resnet --model_depth 18 --resnet_shortcut B --batch_size 128 --n_threads 16 --gpu 0 --test --no_train --no_val --resume_path OUTPUT_MODEL_PATH/save_30.pth
```
