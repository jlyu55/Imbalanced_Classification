import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from collections import Counter
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def get_hist1(logit, label, sub_class, cls, bound, bins=20, file_prefix='', file_suffix='', save=False):
    idx = np.where(label == cls)[0]
    with plt.style.context('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle'):
        fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=False)
        fig.subplots_adjust(bottom=0.2)
        if bound == cls:
            col = '#5A5B9F'
        else:
            col = '#F0C05A'
        ax.hist(logit[bound, idx], density=False, bins=bins, edgecolor='k', color=col, alpha=0.75, linewidth=0.5)
        ax.set_xlabel(f'logit for label {sub_class[bound]}', fontsize=12)
    if save:
        plt.savefig(f'{file_prefix}class{sub_class[cls]} label{sub_class[bound]}{file_suffix}.pdf')
    plt.close()
    return None


def get_2D_hist_stat(logit, label, cls, bound_1, bound_2, bins_1=30, bins_2=30):
    idx = np.where(label == cls)[0]
    x = logit[bound_1, idx]
    y = logit[bound_2, idx]
    hist, xedges, yedges = np.histogram2d(x, y, bins=(bins_1, bins_2), density=True)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    return hist.T, xcenters, ycenters, xedges, yedges


def get_Gauss_contour_stat(logit, label, cls, bound_1, bound_2):
    idx = np.where(label == cls)[0]
    x = logit[bound_1, idx]
    y = logit[bound_2, idx]
    xy = np.vstack([x, y]).T
    mean = np.mean(xy, axis=0)
    cov = np.cov(xy, rowvar=False)
    return mean, cov


def plot_2D_heatmap_with_contour(sub_class, cls, bound_1, bound_2,
                                 hist, xedges, yedges, gauss_mean, gauss_cov,
                                 xlim=None, ylim=None, xpad=0, ypad=0, figsize=(6, 5),
                                 title='Empirical',
                                 cmap=cm.get_cmap('Blues').reversed(), vmax=None,
                                 save=False, file_prefix='', file_suffix=''):
    fig, ax = plt.subplots(figsize=figsize)
    if xlim is None:
        xlim = (np.min(xedges)-xpad, np.max(xedges)+xpad)
    if ylim is None:
        ylim = (np.min(yedges)-ypad, np.max(yedges)+ypad)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    cmap = cm.get_cmap(cmap)
    ax.set_facecolor(cmap(0))
    # Heatmap
    c = ax.pcolormesh(xedges, yedges, hist, cmap=cmap, shading='auto', vmin=0, vmax=vmax)
    fig.colorbar(c, ax=ax, label='Density')
    # Contour plot
    x = np.linspace(*xlim, 200)
    y = np.linspace(*ylim, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack([X, Y])
    normal = multivariate_normal(mean=gauss_mean, cov=gauss_cov)
    Z = normal.pdf(pos)
    ax.contour(X, Y, Z, cmap=cm.get_cmap('Greys'), linewidths=1, linestyles='dashed', alpha=0.3)
    ax.set_title(f'{title} logit distribution of class {sub_class[cls]}')
    ax.set_xlabel(f'logit for label {sub_class[bound_1]}')
    ax.set_ylabel(f'logit for label {sub_class[bound_2]}')
    if save:
        plt.savefig(
            f'{file_prefix}class{sub_class[cls]} label({sub_class[bound_1]},{sub_class[bound_2]}){file_suffix}.pdf')
    return fig, ax


def logisticReg(X, y, tol=1e-8, class_weight=None, solver='lbfgs', max_iter=int(1e+6)):
    clf = LogisticRegression(penalty='l2', tol=tol, fit_intercept=True, class_weight=class_weight,
                             solver=solver, max_iter=max_iter, C=1e+8)
    clf.fit(X, y)
    beta = clf.coef_
    beta0 = clf.intercept_
    return [clf, beta, beta0]


def get_accuracy(y_true, y_pred, Class):
    cm = confusion_matrix(y_true, y_pred, labels=Class)
    acc = cm.diagonal() / cm.sum(axis=1)
    tab = pd.DataFrame(acc, index=Class).transpose()
    tab['tot'] = np.sum(cm.diagonal()) / cm.sum()
    tab['bal'] = np.mean(acc)
    return tab


########################
###  For simulation  ###
########################
def sim_data_multi(n, d, pi, mu, sig=None, seed=2023):
    """ Generate Gaussian Mixture Data """
    K = pi.shape[0]
    if sig is None:
        sig = np.ones(K)
    n_k = np.round(n * pi).astype(int)
    n_k[K-1] = n - np.sum(n_k[:(K-1)])
    X_list = []
    np.random.seed(seed)
    MU = np.random.randn(K, d)
    MU = MU/np.linalg.norm(MU, axis=1)[:, np.newaxis] * mu
    for k in range(K):
        Sig_k = sig[k] * np.identity(d)
        X_k = np.random.multivariate_normal(mean=MU[k, :], cov=Sig_k, size=n_k[k])
        X_list.append(X_k)
    X = np.vstack(X_list)
    y = np.repeat(np.arange(K), n_k)
    print(dict(Counter(y)))
    return [X, y]


def main1(sub_class=(1, 2, 3), C=0, B1=0, B2=1, SIMULATE=False, seed=2023, save=False):
    # Load Data
    if SIMULATE:
        # Simulate Data
        X, y = sim_data_multi(n=50000, d=6000, pi=np.array([0.5, 0.3, 0.2]), mu=4)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
        # Multinomial logistic regression
        clf, beta, beta0 = logisticReg(X_train, y_train, class_weight=None)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        print('Training Accuracy:')
        print(get_accuracy(y_train, y_train_pred, clf.classes_))
        print('Testing Accuracy:')
        print(get_accuracy(y_test, y_test_pred, clf.classes_))
        # Compute logits and save data
        logit_train = beta @ X_train.T + beta0[:, np.newaxis]
        logit_test = beta @ X_test.T + beta0[:, np.newaxis]
        np.save('sim_data/logit_train.npy', logit_train)
        np.save('sim_data/logit_test.npy', logit_test)
        np.save('sim_data/y_train.npy', y_train)
        np.save('sim_data/y_test.npy', y_test)
    else:
        logit_train = np.load('sim_data/logit_train.npy')
        logit_test = np.load('sim_data/logit_test.npy')
        y_train = np.load('sim_data/y_train.npy')
        y_test = np.load('sim_data/y_test.npy')
    print('Training Set:')
    print(dict(Counter(y_train)))
    print('Testing Set:')
    print(dict(Counter(y_test)))
    # 1D Histogram
    get_hist1(logit_train, y_train, sub_class, C, B1, bins=20, file_prefix='CIFAR10 ', file_suffix=' ELD', save=save)
    get_hist1(logit_test, y_test, sub_class, C, B1, bins=20, file_prefix='CIFAR10 ', file_suffix=' TLD', save=save)
    # 2D Histogram
    hist1, _, _, xedges1, yedges1 = get_2D_hist_stat(logit_train, y_train, C, B1, B2)
    hist2, _, _, xedges2, yedges2 = get_2D_hist_stat(logit_test, y_test, C, B1, B2)
    gauss_mean, gauss_cov = get_Gauss_contour_stat(logit_test, y_test, C, B1, B2)
    vmax = hist1.max()
    plot_2D_heatmap_with_contour(sub_class, C, B1, B2, hist1, xedges1, yedges1, gauss_mean, gauss_cov,
                                 file_prefix='CIFAR10 ', file_suffix=' ELD', save=save)
    plot_2D_heatmap_with_contour(sub_class, C, B1, B2, hist2, xedges2, yedges2, gauss_mean, gauss_cov,
                                 file_prefix='CIFAR10 ', file_suffix=' TLD', save=save, title='Testing')
    return hist1, xedges1, yedges1, gauss_mean, gauss_cov, vmax


#######################
###  For real data  ###
#######################
def fit_whitening(x):
    mean = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    U, S, Vh = np.linalg.svd(cov)
    cov_sqrt_inv = np.dot(Vh.T * (S ** (-1 / 2)), U.T)
    return mean, cov_sqrt_inv


def apply_whitening(x, mean, cov_sqrt_inv):
    x_centered = x - mean
    x_whitened = x_centered @ cov_sqrt_inv
    return x_whitened


def main2(sub_class=(1, 2, 3), imb_factor=0.2, C=0, B1=0, B2=1, whitening=True, seed=2024, save=False):
    # Load Data
    DATA_DIR = "../data"
    X = np.load(os.path.join(DATA_DIR, "CIFAR10_ResNet18_pretrain_X_test.npy"))
    y = np.load(os.path.join(DATA_DIR, "CIFAR10_ResNet18_pretrain_y_test.npy"))

    X_sub = X[np.isin(y, sub_class), :]
    y_sub = y[np.isin(y, sub_class)]
    # re-level the labels as 0, 1, 2, ...
    for i in range(len(sub_class)):
        y_sub[y_sub == sub_class[i]] = i

    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.5, random_state=seed, stratify=y_sub)
    SAMPLE_SIZE = y_test.shape[0]
    NUM_OF_CLASS = len(Counter(y_test))

    def get_img_num_per_cls(n=SAMPLE_SIZE, cls_num=NUM_OF_CLASS, imb_type='exp', imb_factor=0.1):
        img_max = n / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num - cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def sim_imb_data(data, cls_num=NUM_OF_CLASS, imb_type='exp', imb_factor=0.1):
        X, y = data
        index = np.array([], dtype=int)
        num_per_cls = get_img_num_per_cls(n=y.shape[0], cls_num=NUM_OF_CLASS, imb_type=imb_type, imb_factor=imb_factor)
        for i in range(cls_num):
            idx_for_cls = np.where(y == i)[0]
            samp_idx = np.sort(np.random.choice(idx_for_cls, size=num_per_cls[i], replace=False))
            index = np.concatenate([index, samp_idx])
        return [X[index, :], y[index]]

    # Generating imbalanced data
    np.random.seed(seed)
    X_imb, y_imb = sim_imb_data([X_train, y_train], imb_type='exp', imb_factor=imb_factor)
    print('Training Set:')
    print(dict(Counter(y_imb)))
    print('Testing Set:')
    print(dict(Counter(y_test)))
    if whitening:
        imb_x_mean, imb_cov_sqrt_inv = fit_whitening(X_imb)
        X_imb = apply_whitening(X_imb, imb_x_mean, imb_cov_sqrt_inv)
        # imb_x_mean, imb_cov_sqrt_inv = fit_whitening(X_test)
        X_test = apply_whitening(X_test, imb_x_mean, imb_cov_sqrt_inv)
    # Multinomial logistic regression
    clf, beta, beta0 = logisticReg(X_imb, y_imb, class_weight=None)
    y_train_pred = clf.predict(X_imb)
    y_test_pred = clf.predict(X_test)
    print('Training Accuracy:')
    print(get_accuracy(y_imb, y_train_pred, clf.classes_))
    print('Testing Accuracy:')
    print(get_accuracy(y_test, y_test_pred, clf.classes_))
    # Calculate logits
    logit_train = beta @ X_imb.T + beta0[:, np.newaxis]
    logit_test = beta @ X_test.T + beta0[:, np.newaxis]
    # 1D Histogram
    get_hist1(logit_train, y_imb, sub_class, C, B1, bins=20, file_prefix='CIFAR10 ', file_suffix=' ELD', save=save)
    get_hist1(logit_test, y_test, sub_class, C, B1, bins=20, file_prefix='CIFAR10 ', file_suffix=' TLD', save=save)
    # 2D Histogram
    hist1, _, _, xedges1, yedges1 = get_2D_hist_stat(logit_train, y_imb, C, B1, B2)
    hist2, _, _, xedges2, yedges2 = get_2D_hist_stat(logit_test, y_test, C, B1, B2)
    gauss_mean, gauss_cov = get_Gauss_contour_stat(logit_test, y_test, C, B1, B2)
    vmax = hist1.max()
    plot_2D_heatmap_with_contour(sub_class, C, B1, B2, hist1, xedges1, yedges1, gauss_mean, gauss_cov,
                                 file_prefix='CIFAR10 ', file_suffix=' ELD', save=save)
    plot_2D_heatmap_with_contour(sub_class, C, B1, B2, hist2, xedges2, yedges2, gauss_mean, gauss_cov,
                                 file_prefix='CIFAR10 ', file_suffix=' TLD', save=save, title='Testing')
    return hist1, xedges1, yedges1, gauss_mean, gauss_cov, vmax
