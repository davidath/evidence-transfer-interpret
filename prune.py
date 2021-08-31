#! /usr/bin/env python

import numpy as np
import sys
import multiprocessing as mp

# Reuse functions
from mrmr import load_data, load_labels, load_mnist, relevance, log 

# K-means
from sklearn.cluster import MiniBatchKMeans as KMeans
# Mutual information (computational solution with KNN)
from sklearn.feature_selection import mutual_info_regression as mir
from sklearn.feature_selection import mutual_info_classif as mif

SEED = 6
SEEDS = 50
FORK_NUM = 6

# Plot relevance comparsion between SAE and evidence transfer

def plot(rels, _rels, width=0.5, out='relevance_plot.png'):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import minmax_scale as mms

    ind = np.arange(len(rels))

    plt.figure()
    plt.bar(ind, mms(rels), width=0.4, label='SAE')
    plt.bar(ind+width, mms(_rels), width=0.4, label='EviT')
    plt.xticks(ind+width / 2, ind)

    plt.legend(loc='best')

    plt.savefig(out)

# Incrementally prune latent features and plot ACC/NMI

def prune_plot(lat, _lat, rels, _rels, labels, out='relevance_plot.png', width=0.5):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    lst = [(i, rels[i]) for i in range(len(rels))]
    lst2 = [(i, _rels[i]) for i in range(len(_rels))]

    lst = sorted(lst, key=lambda k: k[1])
    lst2 = sorted(lst2, key=lambda k: k[1])
    
    feat_idx = range(10)
    baseline = prune_predict(lat, labels)
    pruned = []

    for x, y in lst:
        try:
            _feat_idx = [i for i in feat_idx if i != x]
            pruned.append(prune_predict(lat[:, _feat_idx], labels))
            feat_idx = _feat_idx
        except ValueError:
            pass

    feat_idx2 = range(10)
    pruned2 = []

    baseline2 = prune_predict(_lat, labels)

    for x, y in lst2:
        try:
            _feat_idx2 = [i for i in feat_idx2 if i != x]
            pruned2.append(prune_predict(_lat[:, _feat_idx2], labels))
            feat_idx2 = _feat_idx2
        except ValueError:
            pass

    #  print [baseline[0]]+[x for x,y in pruned]
    #  print [baseline2[0]]+[x for x,y in pruned2]

    ind = np.arange(lat.shape[1])

    fig, ax = plt.subplots(2,1)
    ax[0].bar(ind, [baseline[0]]+[x for x,y in pruned], width=0.4, label='ACC')
    ax[0].bar(ind+width, [baseline[1]]+[y for x,y in pruned], width=0.4, label='NMI')
    ax[0].plot(ind, [baseline[0]]+[baseline[0] for x,y in pruned], 'k')
    ax[0].plot(ind, [baseline[1]]+[baseline[1] for x,y in pruned], 'k--')
    #  ax[0].set_xticks(ind + width / 2, ind)
    ax[0].set_title('SAE')

    ax[1].bar(ind, [baseline2[0]]+[x for x,y in pruned2], width=0.4, label='ACC')
    ax[1].bar(ind+width, [baseline2[1]]+[y for x,y in pruned2], width=0.4, label='NMI')
    ax[1].plot(ind, [baseline2[0]]+[baseline2[0] for x,y in pruned], 'k')
    ax[1].plot(ind, [baseline2[1]]+[baseline2[1] for x,y in pruned], 'k--')
    #  ax[1].set_xticks(ind+width / 2, ind)
    ax[1].set_title('Evi-T')

    plt.legend(loc='best')

    plt.savefig(out)


# Log script running instructions

def _help():
    log('', 'Help:')

# Unsupervised clustering accuracy from DEC
# (https://arxiv.org/pdf/1511.06335.pdf)

def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in xrange(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w

# Worker function to make multiple seeded runs for MI estimation
def do_parallel(z, y, r, metric):
    return metric(z, y, random_state=r)

# Unpacking values to pass on worker
def unpack(args):
    return do_parallel(*args)

# Compute relevance metric for latent representations 
# before/after evidence tranfer (Mutual information / F-test 
# between features and class labels)

def relevance(z, y, metric):
    try:
        if metric == mif or metric == mir:
            # Multi core MI estimation
            rand = np.random.randint(0, 1e4, size=SEEDS)
            pool = mp.Pool(processes=FORK_NUM)
            res = pool.map(unpack, [(z, y, r, metric) for r in rand])
            res = np.array(res)
            pool.close()
            pool.join()
        else:
            # Single core estimation of any metric (no random seeds)
            res = metric(z, y)
        return res
    # F-test
    except AttributeError:
        res = metric(z, y)
        return res[0]


def prune_predict(lat, labels, nc=10, init=100):
    from sklearn.metrics import normalized_mutual_info_score as NMI
    np.random.seed(SEED)
    km = KMeans(n_clusters=nc, n_init=init, random_state=SEED)
    pred = km.fit_predict(lat)
    return cluster_acc(pred, labels)[0], NMI(labels, pred)


def get_relevance(lat, labels):
    # Get relevance
    rel = relevance(lat, labels, mif)
    _rel = []
    for r in range(rel.shape[1]):
        feat_mean = rel[:, r].mean()
        _rel.append(feat_mean)
    _rel = np.array(_rel)
    return _rel

# Ranking correlation

def corr(rels, _rels, evi_lat_path):
    # Spearman's r
    from scipy.stats import spearmanr as spr
    # Kendalls T
    from scipy.stats import kendalltau as kdt
    # Cosine distance
    from scipy.spatial.distance import cosine as cd
    # Make ranking tupples
    lst = [(i, rels[i]) for i in range(len(rels))]
    lst2 = [(i, _rels[i]) for i in range(len(_rels))]
    # Sort tupples
    lst = sorted(lst, key=lambda k: k[1])
    lst2 = sorted(lst2, key=lambda k: k[1])
    # Keep ranks
    lst = [i[0] for i in lst]
    lst2 = [i[0] for i in lst2]

    # Calculate simple variation
    var = [p for p, i in enumerate(lst) if lst[p] != lst2[p]]
    var = len(var) / float(len(lst))

    log(spr(lst, lst2)[0], label='@'+evi_lat_path.split('_')[1]+'@ Correlation !R!:')
    log(kdt(lst, lst2)[0], label='@'+evi_lat_path.split('_')[1]+'@ Correlation !T!:')
    log(cd(lst, lst2), label='@'+evi_lat_path.split('_')[1]+'@ Correlation !CD!:')
    log(var, label='@'+evi_lat_path.split('_')[1]+'@ Correlation !Var!:')


def build_main():
    # Run flow controller
    # MNIST
    try:
        if len(sys.argv[1].split('mnist')) > 1:
            def main(lat_path, plot_flag, evi_lat_path, label_path):
                # Load latent samples
                Z = load_data(lat_path)
                _Z = load_data(evi_lat_path)

                # Load random var c (classification task labels)
                y = load_labels(label_path)

                res = get_relevance(Z, y)

                _res = get_relevance(_Z, y)

                if plot_flag == 'plot':
                    plot(res, _res, out=evi_lat_path.split('_')[1] + '.png')
                elif plot_flag == 'corr':
                    corr(res, _res, evi_lat_path)
                else:
                    prune_plot(Z, _Z, res, _res, y, out=evi_lat_path.split('_')[1] + '.png')
        else:
            def main(lat_path, label_path, ppath, plot_flag, evi_lat_path):
                # Load latent samples
                Z = load_data(lat_path)
                _Z = load_data(evi_lat_path)

                # Load random var c (classification task labels)
                y = load_labels(label_path, ppath)

                res = get_relevance(Z, y)

                _res = get_relevance(_Z, y)

                if plot_flag == 'plot':
                    plot(res, _res, out=evi_lat_path.split('_')[1] + '.png')
                elif plot_flag == 'corr':
                    corr(res, _res, evi_lat_path)
                else:
                    prune_plot(Z, _Z, res, _res, y, out=evi_lat_path.split('_')[1] + '.png')
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
