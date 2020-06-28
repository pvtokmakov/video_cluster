import pandas as pd
import gzip
import numpy as np
import pickle
import warnings
import glob
from sketching import *
from os.path import isfile, join
import os
import torch
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="IDT video inference")
    parser.add_argument("--category", help="Category to process.")
    parser.add_argument("--model_path", help="Path to FV models.")
    parser.add_argument("--videos_path", help="Path to videos.")
    parser.add_argument("--boxes_path", help="Path to boxes.")
    parser.add_argument("--out_path", help="Output path.")
    return parser


def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors/features.

    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors

    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.

    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.

    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """

    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
            - Q_xx_2
            - Q_sum * gmm.means_ ** 2
            + Q_sum * gmm.covariances_ + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_mu.flatten(), d_sigma.flatten()))


def power_normalize(xx, alpha=0.5):
    """Computes a alpha-power normalization for the matrix xx."""
    return np.sign(xx) * np.abs(xx) ** alpha


def L2_normalize(xx):
    """L2-normalizes each row of the data xx."""
    Zx = np.sum(xx * xx, 1)
    xx_norm = np.divide(xx, np.sqrt(Zx[:, np.newaxis]))
    xx_norm[np.isnan(xx_norm)] = 0
    return xx_norm


def compute_fv(filename, gmm, pca, sdense_fv):
    # use '\s+' for the \t and \n
    df = pd.read_table(gzip.open(filename), sep='\s+', header=None)
    df = df.iloc[:, 10:436]

    # turn pandas dataframe into array
    df_array = df.values
    if np.any(np.isnan(df_array)):
        return None
    # use stored PCA

    if df_array.shape[0] > 3000000:
        print("Dropping tracks to save memory")
        df_array = df_array[:3000000]

    df_array = pca.transform(df_array)
    # get the fisher vector for each video sequence
    fv = fisher_vector(df_array, gmm)

    fv = power_normalize(fv, alpha=0.5)
    fv = np.expand_dims(fv, axis=0)
    fv = L2_normalize(fv)

    SKETCH_FLOAT = 0
    SKETCH_DIM_feat = 2000

    FEAT_DIM_fv = fv.shape[1]

    fv = sketch_batch(torch.from_numpy(fv), sdense_fv, SKETCH_DIM_feat, FEAT_DIM_fv, SKETCH_FLOAT)

    return fv


def process_category(model_path, videos_path, boxes_path, out_path, category):
    print(category)
    pca = pickle.load(open(join(model_path, "pca_model.sav"), 'rb'))
    gmm = pickle.load(open(join(model_path, "gmm_diag_model.sav"), 'rb'))
    sdense_fv = pickle.load(open(join(model_path, "sketching_proj.sav"), 'rb'))

    box_path = join(boxes_path, category)
    boxes = [f for f in os.listdir(box_path) if isfile(join(box_path, f))]

    vid_path = join(videos_path, category)
    vids = [f for f in os.listdir(vid_path) if isfile(join(vid_path, f))]

    out_path = join(out_path, category)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    count = 0
    temp_path = "/scratch/ptokmako/IDT_features_temp"
    for vid in vids:
        vid_name = vid.split(".")[0]
        
        if (vid_name + ".bb") not in boxes:
            print("Boxes not found for %s !!!!!!!!!!" % vid_name)
            continue
        count += 1
        if os.path.exists(out_path + "%s.dat" % vid_name):
            continue
        print("%d/%d %s" % (count, len(vids), vid_name))

        sz = 0
        filename = "%s/%s.gz" % (temp_path, category + "_" + vid_name)
        attempts = 0
        while sz < 100 and attempts < 5:
            stream = os.popen('sh src/idt/idt.sh "%s" "%s" "%s" %s' % (join(vid_path, vid), join(box_path, vid_name + ".bb"),
                                                            category + "_" + vid_name, temp_path))
            output = stream.read()
            print(output)
            sz = os.path.getsize(filename)
            attempts += 1

        if sz < 100:
            print("Could not process video!!!!!!!!!!!!!!")
            continue

        fv = compute_fv(filename, gmm, pca, sdense_fv)
        if fv is not None:
            torch.save(fv, join(out_path, vid_name + ".dat"))
        else:
            print("NaNs in IDT\n")

        stream = os.popen('rm -f "%s/%s.gz"' % (temp_path, category + "_" + vid_name))
        output = stream.read()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = get_parser().parse_args()
    model_path = args.model_path
    videos_path = args.videos_path
    boxes_path = args.boxes_path
    out_path = args.out_path
    category = args.category
    
    process_category(model_path, videos_path, boxes_path, out_path, category)



