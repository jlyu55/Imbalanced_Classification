import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.stats import norm
from collections import Counter


# Only applicable for multiclass data with EQUAL size
def get_img_num_per_cls(n, cls_num, imb_type='exp', imb_factor=0.1):
    img_max = n / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num - cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls


def sim_imb_data(data, imb_type='exp', imb_factor=0.1):
    X, y = data
    num_class = len(np.unique(y))
    index = np.array([], dtype=int)
    print(num_class, y.shape[0])
    num_per_cls = get_img_num_per_cls(n=y.shape[0], cls_num=num_class, imb_type=imb_type, imb_factor=imb_factor)
    for i in range(num_class):
        idx_for_cls = np.where(y == i)[0]
        samp_idx = np.sort(np.random.choice(idx_for_cls, size=num_per_cls[i], replace=False))
        index = np.concatenate([index, samp_idx])
    return [X[index, :], y[index]]


# Binary classification
# Minority: y = 1, Majority: y = 0
def sim_imb_data2(data, pai=None, random_state=2024):
    X, y = data
    if pai is None:
        return [X, y]
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    [n, n0, n1] = [y.shape[0], idx_0.shape[0], idx_1.shape[0]]
    np.random.seed(random_state)
    if pai < n1/n:
        samp_idx = np.sort(np.random.choice(idx_1, size=int(n0*pai/(1-pai)), replace=False))
        index = np.concatenate([idx_0, samp_idx])
    else:
        samp_idx = np.sort(np.random.choice(idx_0, size=int(n1*(1-pai)/pai), replace=False))
        index = np.concatenate([samp_idx, idx_1])
    return [X[index, :], y[index]]


def logisticReg(X, y, tol=1e-8, class_weight=None, solver='lbfgs', max_iter=int(1e+6), standard=False):
    clf = LogisticRegression(penalty=None, tol=tol, fit_intercept=True, class_weight=class_weight,
                             solver=solver, max_iter=max_iter)
    clf.fit(X, y)
    beta = clf.coef_[0]
    beta0 = clf.intercept_[0]
    if standard:
        beta_norm = np.linalg.norm(beta)
        beta, beta0 = beta/beta_norm, beta0/beta_norm
    return [clf, beta, beta0]


def SVM(X, y, C=1, shrinking=True, tol=1e-8, class_weight=None, standard=True):
    clf = SVC(C=C, kernel='linear', shrinking=shrinking, tol=tol, class_weight=class_weight)
    clf.fit(X, y)
    beta = clf.coef_[0]
    beta0 = clf.intercept_[0]
    kappa = 1
    if standard:
        beta_norm = np.linalg.norm(beta)
        beta, beta0, kappa = beta/beta_norm, beta0/beta_norm, kappa/beta_norm
    return [clf, beta, beta0, kappa]


def get_accuracy(y_true, y_pred, Class):
    cm = confusion_matrix(y_true, y_pred, labels=Class)
    acc = cm.diagonal()/cm.sum(axis=1)
    tab = pd.DataFrame(acc, index=Class).transpose()
    tab['tot'] = np.sum(cm.diagonal())/cm.sum()
    tab['bal'] = np.mean(acc)
    return tab


def get_results(y_true, y_pred, Class, figsize, title, plot=True):
    acc = get_accuracy(y_true, y_pred, Class)
    if plot:
        cm = confusion_matrix(y_true, y_pred, labels=Class)
        cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Class)
        fig, ax = plt.subplots(figsize=figsize)
        cmp.plot(ax=ax)
        plt.title(title)
        plt.show()
        print(acc)
    return acc


def fit_whitening(x, rank=None):
    n, d = x.shape
    if rank is None:
        rank = d
    mean = np.mean(x, axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    perm = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[perm], eigvecs[:, perm]
    if rank > 0:
        eigvals_smooth = np.concatenate((eigvals[:rank], np.ones(d-rank) * np.mean(eigvals[rank:]))) if rank < d else eigvals
        cov_sqrt_inv = eigvecs @ np.diag(np.sqrt(1 / eigvals_smooth)) @ eigvecs.T 
    else:
        cov_sqrt_inv = np.eye(d)
    return mean, cov_sqrt_inv


def apply_whitening(x, mean, cov_sqrt_inv):
    x_centered = x - mean
    x_whitened = x_centered @ cov_sqrt_inv
    return x_whitened


def fit_gaussian(margin, y):
    assert len(margin) == len(y)
    params = {}
    for k in range(2):
        params[k] = [np.mean(margin[y==k]), np.std(margin[y==k])]
    return params


def LR_logit_experiment(X, y, save_dir, pai = 0.1, test_size=0.5, random_state=2024, classes=(0, 1),
                        whitening_rank=0, overlay_fitted_test_Gaussian=False, title='ELD',
                        save=False, fig_size=(6, 4), binwidth0=1, binwidth1=1, bins0=None, bins1=None,
                        legend=False, title_font=20, text_font=16
                        ):
    # Split data and print info
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    # Create imbalanced train dataset
    X_imb, y_imb = sim_imb_data2([X_train, y_train], pai=pai, random_state=random_state)
    print("Imbalanced training Set:", dict(Counter(y_imb)))
    print("pi =", f"{dict(Counter(y_imb))[1]/np.shape(y_imb)[0]:.3f}")
    print("n =", f"{y_imb.shape[0]}")

    # Apply optional whitening to train features
    imb_x_mean, imb_cov_sqrt_inv = fit_whitening(X_imb, rank=whitening_rank)
    X_imb_white = apply_whitening(X_imb, imb_x_mean, imb_cov_sqrt_inv)
    X_test_white = apply_whitening(X_test, imb_x_mean, imb_cov_sqrt_inv)

    # (Binary) logistic regression
    clf, beta, beta0 = logisticReg(X_imb_white, y_imb, class_weight=None)

    # Get logits
    logits = X_imb_white @ beta + beta0
    logits_test = X_test_white @ beta + beta0
    params = fit_gaussian(logits_test.flatten(), y_test)

    logit1 = logits[np.where(y_imb == 1)]
    logit0 = logits[np.where(y_imb == 0)]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    with plt.style.context('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle'):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=fig_size, dpi=100, constrained_layout=True)
        fig.suptitle(title, fontweight='bold', fontsize=title_font)
        cols = ['#5A5B9F', '#D94F70', '#009473', '#F0C05A', '#7BC4C4', '#FF6F61']

        if bins1 is None:
            bins1 = np.arange(min(logit1), max(logit1) + binwidth1, binwidth1)
        if bins0 is None:
            bins0 = np.arange(min(logit0), max(logit0) + binwidth0, binwidth0)

        # axs.set_title(r'Gaussian Mixture Model')
        axs.hist(logit1, density=True, bins=bins1, edgecolor=None, alpha=0.30, color=cols[0],
                 label=r'Minority ELD')
        axs.hist(logit0, density=True, bins=bins0, edgecolor=None, alpha=0.40, color=cols[3],
                 label=r'Majority ELD')

        # theoretical density for empirical logit
        def get_density(xmin, xmax, loc, scale, **kwargs):
            xx = np.linspace(xmin, xmax, 1000)
            axs.plot(xx, norm.pdf(xx, loc=loc, scale=scale), **kwargs)
            return None

        if overlay_fitted_test_Gaussian:
            kappa0 = min(-logit0)
            kappa1 = min(logit1)
            get_density(xmin=kappa1, xmax=np.max(logit1), loc=params[1][0], scale=params[1][1],
                        label='Minority TLD', color=cols[0])
            get_density(xmin=np.maximum(0, np.min(logit0)), xmax=kappa1, loc=params[1][0], scale=params[1][1],
                        linestyle=':', color=cols[0])
            get_density(xmin=np.min(logit0), xmax=-kappa0, loc=params[0][0], scale=params[0][1],
                        label='Majority TLD', color=cols[3])
            get_density(xmin=-kappa0, xmax=np.minimum(0, np.max(logit1)), loc=params[0][0], scale=params[0][1],
                        linestyle=':', color=cols[3])

        # decision boundary
        axs.axvline(x=0, color='red', linestyle='--', label='Decision boundary')
        if legend:
            axs.legend(fontsize=text_font+2)
        axs.set_xlabel(r'logit', fontsize=text_font+2)
        axs.tick_params(axis='x', labelsize=text_font)
        axs.tick_params(axis='y', labelsize=text_font)

    if save:
        plt.savefig(save_dir + f'classes_{classes[0]}_{classes[1]}_whitening_rank_{whitening_rank}'
                    + f'pi_{pai:.3f}_{random_state}' + '.pdf')
    else:
        plt.show()
    return None
