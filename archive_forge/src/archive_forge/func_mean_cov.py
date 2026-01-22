import numpy as np
import scipy.misc as sc
import scipy.signal
import scipy.io
def mean_cov(X):
    n, p = X.shape
    m = X.mean(axis=0)
    cx = X - m
    S = dgemm(1.0 / (n - 1), cx.T, cx.T, trans_a=0, trans_b=1)
    return (cx, m, S.T)