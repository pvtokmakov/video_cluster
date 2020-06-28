from src.datasets.kinetics import load_annotation_data, get_video_names_and_annotations, load_value_file
import os
import torch
import json
import argparse
from os.path import join
import numpy as np
from src.objectives.localagg import run_kmeans_multi_gpu, run_kmeans


DEFAULT_KMEANS_SEED = 1234


def get_parser():
    parser = argparse.ArgumentParser(description="IDT video inference")
    parser.add_argument("--k", type=int, help="Number of clusters.")
    parser.add_argument("--num_c", type=int, help="Number of clusterings.")
    parser.add_argument("--frames_path", help="Path to Kinetics frames.")
    parser.add_argument("--annotation_path", help="Path to Kinetics annotation.")
    parser.add_argument("--fv_path", help="Path to Fisher vectors.")
    parser.add_argument("--clusters_path", help="Path to save cluster.")
    parser.add_argument("--processed_annotation_path", help="Path to output annotation file.")
    parser.add_argument('--gpu', nargs='*', help='GPU id')
    return parser


def compute_clusters(data, k, gpu_devices):
    pred_labels = []
    data_npy = data.cpu().detach().numpy()
    data_npy = np.float32(data_npy)
    for k_idx, each_k in enumerate(k):
        # cluster the data

        if len(gpu_devices) == 1: # single gpu
            I, _ = run_kmeans(data_npy, each_k, seed=k_idx + DEFAULT_KMEANS_SEED,
                              gpu_device=gpu_devices[0])
        else: # multigpu
            I, _ = run_kmeans_multi_gpu(data_npy, each_k, seed=k_idx + DEFAULT_KMEANS_SEED, gpu_device=gpu_devices)

        clust_labels = np.asarray(I)
        pred_labels.append(clust_labels)
    pred_labels = np.stack(pred_labels, axis=0)
    pred_labels = torch.from_numpy(pred_labels).long()

    return pred_labels


if __name__ == "__main__":
    args = get_parser().parse_args()
 
    gpu_devices = []
    if args.gpu:
        ids_list = ''
        for i in range(len(args.gpu)):
            ids_list += args.gpu[i] + ','
            gpu_devices.append(int(args.gpu[i]))
        ids_list = ids_list[:-1]

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ids_list

    frames_path = args.frames_path
    annotation_path = args.annotation_path
    fv_path = args.fv_path
    k = args.k
    n_clusters = args.num_c

    data = load_annotation_data(annotation_path)

    video_names, annotations = get_video_names_and_annotations(data, "training")

    count_valid = 0
    count_missing = 0
    fvs = []
    database = {}
    labels = set([])
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        vid_key = video_names[i].split("/")[-1]
        vid_label = video_names[i].split("/")[-2].replace("_", " ")

        video_path = os.path.join(frames_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        count_valid += 1

        fv_vid_path = os.path.join(fv_path, video_names[i]) + ".dat" 
        if not os.path.exists(fv_vid_path):
            count_missing += 1
            continue
        else:
            fv = torch.load(fv_vid_path)

        value = {}
        value['subset'] = 'training'
        value['annotations'] = {}
        value['annotations']['label'] = vid_label
        database[vid_key] = value

        labels.add(vid_label)

        fvs.append(fv.cpu().squeeze())

    for key, value in data['database'].items():
        this_subset = value['subset']
        if (this_subset == 'validation' and (value['annotations']['label'] in labels)) or this_subset == 'testing':
            database[key] = value

    print("%d missing out of %d\n" % (count_missing, count_valid))

    fvs = torch.stack(fvs)

    k = [k for _ in range(n_clusters)]

    cluster_labels = compute_clusters(fvs, k, gpu_devices)

    os.mkdir(join(args.clusters_path, "checkpoints"))
    torch.save({'cluster_labels': cluster_labels}, join(args.clusters_path, "checkpoints/checkpoint.pth.tar"))

    out = {}
    out['labels'] = list(labels)
    out['database'] = database

    with open(args.processed_annotation_path, 'w') as dst_file:
        json.dump(out, dst_file)
