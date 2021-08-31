#! /usr/bin/env python

import numpy as np
import os
import sys
import tensorflow as tf
import multiprocessing as mp

import itertools

# Suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Mutual information (computational solution with KNN)
from sklearn.feature_selection import mutual_info_regression as mir
from sklearn.feature_selection import mutual_info_classif as mif
# ANOVA F-value
from sklearn.feature_selection import f_classif as annof
from sklearn.feature_selection import f_regression as annor
# Pearson correlation
from scipy.stats import pearsonr as pcorr

# MI random state seed numbers
SEEDS = 50

# Number of forks for MI estimation
FORK_NUM = 6

# Logging messages such as loss,loading,etc.

def log(s, label='INFO'):
    from datetime import datetime
    sys.stdout.write(label + ' ' + str(s) + '\n')
    sys.stdout.flush()

# Download / Extract mnist data

def load_mnist(val_size=0, path='/tmp/mnist/'):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(
        path, validation_size=val_size)
    return mnist

# Loading labels (Classification task labels)

def load_labels(dpath='', ppath='mnist/MNIST_perm.npy'):
    if dpath == '':
        data = load_mnist()
        perm = np.load(ppath)
        y = np.concatenate((data.train.labels, data.test.labels))
        y = y[perm]
    else:
        data = np.load(dpath)
        perm = np.load(ppath)
        y = data[perm]
    return y

# Load data (Encoder output)

def load_data(dpath):
    return np.load(dpath)

# Worker function to make multiple seeded runs for MI estimation
def do_parallel(z, y, r, metric):
    return metric(z, y, random_state=r).mean()

# Unpacking values to pass on worker
def unpack(args):
    return do_parallel(*args)

# Compute relevance metric for latent representations 
# before/after evidence tranfer (Mutual information / F-test 
# between features and class labels)

def relevance(z, _z, y, metric, run_flag='multi'):
    try:
        if run_flag == 'multi':
            if metric == mif or metric == mir:
                # Multi core MI estimation
                rand = np.random.randint(0, 1e4, size=SEEDS)
                pool = mp.Pool(processes=FORK_NUM)
                evi_t = pool.map(unpack, [(_z, y, r, metric) for r in rand])
                evi_t = np.array(evi_t)
                pool.close()
                pool.join()
            else:
                # Single core estimation of any metric (no random seeds)
                evi_t = metric(_z, y)
            return 0, evi_t.mean()
        else:
            if metric == mif or metric == mir:
                # Multi core MI estimation
                rand = np.random.randint(0, 1e4, size=SEEDS)
                pool = mp.Pool(processes=FORK_NUM)
                sae = pool.map(unpack, [(z, y, r, metric) for r in rand])
                sae = np.array(sae)
                pool.close()
                pool.join()
            else:
                # Single core estimation of any metric (no random seeds)
                sae = metric(z, y)
            return sae.mean(), 0
    # F-test
    except AttributeError:
        sae = metric(z, y)
        evi_t = metric(_z, y)
        return sae[0].mean(), evi_t[0].mean()

# Compute redundancy metric for latent representations 
# before/after evidence tranfer (Mutual information / F-test 
# between features)

def redundancy(z, _z, y, metric, run_flag='multi'):
    if run_flag == 'multi':
        sample_num = len(_z)
        _red = []
        _combs = itertools.combinations(range(_z.shape[1]), 2)
        for c, i in enumerate(_combs):
            # Multi core MI estimation
            if metric == mif or metric == mir:
                rand = np.random.randint(0, 1e4, size=SEEDS)
                pool = mp.Pool(processes=FORK_NUM)
                _temp = pool.map(unpack, [(_z[:, i[0]].reshape(sample_num, 1),
                        _z[:, i[1]].reshape(sample_num), r, metric) for r in rand])
                pool.close()
                pool.join()
                _temp = np.array(_temp)
                _red.append(_temp.mean())
            # Single core estimation of any metric (no random seeds)
            else:
                _red.append(metric(_z[:, i[0]].reshape(sample_num, 1),
                                  _z[:, i[1]].reshape(sample_num))[0])
        _red = np.array(_red)
        return 0, _red.mean()
    else:
        sample_num = len(z)
        red = []
        combs = itertools.combinations(range(z.shape[1]), 2)
        for i in combs:
            # Multi core MI estimation
            if metric == mif or metric == mir:
                rand = np.random.randint(0, 1e4, size=SEEDS)
                pool = mp.Pool(processes=FORK_NUM)
                temp = pool.map(unpack, [(z[:, i[0]].reshape(sample_num, 1),
                    z[:, i[1]].reshape(sample_num), r, metric) for r in rand])
                pool.close()
                pool.join()
                temp = np.array(temp)
                red.append(temp.mean())
            # Single core estimation of any metric (no random seeds)
            else:
                red.append(metric(z[:, i[0]].reshape(sample_num, 1),
                                  z[:, i[1]].reshape(sample_num))[0])
        red = np.array(red)
        return red.mean(), 0


# Compute pearson correlation between latent representations
# before/after evidence tranfer and class labels

def label_correlation(z, _z, y, metric):
    corr = []
    # Iterate between features
    for i in range(z.shape[1]):
        corr.append(metric(z[:, i], y))
    corr = np.array(corr)[:, 0]

    _corr = []
    # Iterate between features
    for i in range(_z.shape[1]):
        _corr.append(metric(_z[:, i], y))
    _corr = np.array(_corr)[:, 0]

    return corr.mean(), _corr.mean()



# Compute pearson correlation between each feature of latent representations
# before/after evidence transfer

def feature_correlation(z, _z, y, metric):
    sample_num = len(z)
    red = []
    combs = itertools.combinations(range(z.shape[1]), 2)
    for i in combs:
        red.append(metric(z[:, i[0]].reshape(sample_num),
                          z[:, i[1]].reshape(sample_num))[0])
    red = np.array(red)

    sample_num = len(_z)
    _red = []
    _combs = itertools.combinations(range(_z.shape[1]), 2)
    for i in _combs:
        _red.append(metric(_z[:, i[0]].reshape(sample_num),
                          _z[:, i[1]].reshape(sample_num))[0])
    _red = np.array(_red)

    return red.mean(), _red.mean()

# Call metric functions
def metric_logs(Z, _Z, y, evi_lat_path, run_flag='multi'):
    # Relevance (MI)
    rel_mi = relevance(Z, _Z, y, mif, run_flag)
    log(rel_mi, label='@'+evi_lat_path.split('_')[1]+'@ Relevance !MI!:')
    # Relevance (F-test)
    rel_ft = relevance(Z, _Z, y, annof, run_flag)
    log(rel_ft, label='@'+evi_lat_path.split('_')[1]+'@ Relevance !F-test!:')
    # Correlation fi,c
    rel_corr = label_correlation(Z, _Z, y, pcorr)
    log(rel_corr, label='@'+evi_lat_path.split('_')[1]+'@ Relevance !P-Correlation!:')

    # Redundancy (MI)
    red_mi = redundancy(Z, _Z, y, mir, run_flag)
    log(red_mi, label='@'+evi_lat_path.split('_')[1]+'@ Redundancy !MI!:')
    # Redundancy (F-test)
    red_ft = redundancy(Z, _Z, y, annor, run_flag) 
    log(red_ft, label='@'+evi_lat_path.split('_')[1]+'@ Redundancy !F-test!:')
    # Correlation fi, fj
    red_corr = feature_correlation(Z, _Z, y, pcorr)
    log(red_corr, label='@'+evi_lat_path.split('_')[1]+'@ Redundancy !P-Correlation!:')

# Print run arguments

def _help():
    log('args: <mnist/initial_z> <single/multi> <mnist/evidence_transfer_z1> ...'+
            '<mnist/evidence_transfer_zn>', 'HELP (MNIST):')
    log('args: <path/initial_z> <path/labels> <path/random_permutation>'+
            '<path/evidence_transfer_z1> <single/multi> ... <path/evidence_transfer_zn>',
            'HELP (Other datasets):')

def build_main():
    # Run flow controller
    # MNIST
    try:
        if len(sys.argv[1].split('mnist')) > 1:
            def main(lat_path, run_flag, evi_lat_path, label_path):
                # Load latent samples
                Z = load_data(lat_path)
                _Z = load_data(evi_lat_path)

                # Load random var c (classification task labels)
                y = load_labels(label_path)

                # Print results
                metric_logs(Z, _Z, y, evi_lat_path, run_flag)
        else:
            def main(lat_path, label_path, ppath, run_flag, evi_lat_path):
                # Load latent samples
                Z = load_data(lat_path)
                _Z = load_data(evi_lat_path)

                # Load random var c (classification task labels)
                y = load_labels(label_path, ppath)

                # Print results
                metric_logs(Z, _Z, y, evi_lat_path, run_flag)
        return main
    except IndexError:
        raise IndexError

if __name__ == "__main__":
    try:
        main = build_main()
        # MNIST
        if len(sys.argv[1].split('mnist')) > 1:
            for f in sys.argv[3:]:
                main(sys.argv[1], sys.argv[2], f, '')
        else:
            # Other datasets
            for f in sys.argv[5:]:
                main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], f)
    except IndexError:
        _help()
