#!/usr/bin/env bash

export PATH=/home/ptokmako/src/opencv/lib:/home/ptokmako/miniconda2/bin:/home/ptokmako/src/ffmpeg/lib:/home/ptokmako/src/ffmpeg/include:/home/ptokmako/miniconda2/envs/idt/lib:/home/ptokmako/miniconda2/envs/idt/include:/opt/cuda/9.1/bin:/home/ptokmako/torch/install/bin:/home/ptokmako/miniconda2/envs/idt/bin:/home/ptokmako/miniconda2/condabin:/opt/gcc49/bin:/opt/openmpi/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/ganglia/bin:/opt/ganglia/sbin:/opt/rocks/bin:/opt/rocks/sbin:/home/ptokmako/bin
export LD_LIBRARY_PATH=/home/ptokmako/src/opencv/lib/:/home/ptokmako/src/ffmpeg/lib/:/home/ptokmako/src/ffmpeg/include/:/home/ptokmako/miniconda2/envs/idt/lib:/home/ptokmako/miniconda2/envs/idt/include:/opt/cuda/9.1/lib64:/opt/cuda/9.1/lib:/home/ptokmako/torch/install/lib:/opt/openmpi/lib:/home/ptokmako/local/readline-8.0/lib

source ~/miniconda2/etc/profile.d/conda.sh

conda activate idt

/home/ptokmako/src/improved_trajectory_release/release/DenseTrackStab $1 -H $2 | gzip > $4/$3.gz
