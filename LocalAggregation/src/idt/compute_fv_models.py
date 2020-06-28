import shutil, random, os
import pandas as pd
import numpy as np
import gzip
from sklearn import decomposition
import pickle
from sklearn.mixture import GaussianMixture
from sketching import *
from extract_idt import fisher_vector
import argparse
from os.path import join


def get_parser():
    parser = argparse.ArgumentParser(description="FV model estimation")
    parser.add_argument("--idt_path", help="Path to precomputed IDTs.")
    return parser


if __name__ == "__main__":
    # define an empty numpy array for concatenating features
    all_features = np.array([])

    args = get_parser().parse_args()
    dirpath = args.idt_path

    counter = 0
    num_feats = 500

    filenames = os.listdir(dirpath)
    for fname in filenames:
        if fname.endswith(".gz"):
            srcpath = os.path.join(dirpath, fname)
            print("concatenating features from: ", srcpath)
            counter = counter + 1
            df = pd.read_table(gzip.open(srcpath), sep='\s+', header=None)

            # turn pandas dataframe into array
            df_array = np.round(df.values, decimals=3)

            array_sum = np.sum(df_array)
            array_has_nan = np.isnan(array_sum)
            if (array_has_nan):
                continue

            if df_array.shape[0] < num_feats:
                print('less than %d' % num_feats)
            else:
                idx = np.random.randint(df_array.shape[0], size=num_feats)
                df_array = df_array[idx, :]
            
            # concatenate all the features
            print('stack feature vectors...', counter)
            all_features = np.vstack([all_features, df_array]) if all_features.size else df_array
            print('Done!-----------------------------')

    features = all_features[:, 10:436]

    trajectories = pd.DataFrame(features)
    print('The feature dimension after random sampling is: ', trajectories.shape)
    print(trajectories.describe())

    pca = decomposition.PCA(0.90)
    pca_features = pca.fit_transform(trajectories)

    print(pca_features.shape)

    filename = join(dirpath, 'pca_model.sav')
    pickle.dump(pca, open(filename, 'wb'))

    K = 256
    gmm = GaussianMixture(n_components=K, covariance_type='diag', n_init=2, max_iter=200)

    print("Start the GMM estimation...")
    gmm.fit(pca_features)
    print("A GMM estimation has been finished!")

    filename = join(dirpath, 'gmm_diag_model.sav')
    pickle.dump(gmm, open(filename, 'wb'))

    SKETCH_DIM_feat = 2000
    fv = fisher_vector(pca_features[0, :], gmm)
    print(fv.shape)

    d_fv = fv.shape[0]
    h_fv = choose_h_sk_mat(SKETCH_DIM_feat, d_fv)
    s_fv = choose_s_sk_mat(2, d_fv)
    sdense_fv = create_s_dense(h_fv, s_fv)

    filename = join(dirpath, 'sketching_proj.sav')
    pickle.dump(sdense_fv, open(filename, 'wb'))
