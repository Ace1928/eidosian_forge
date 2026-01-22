from os.path import dirname
from os.path import join
import numpy as np
import scipy.fftpack
import scipy.io
import scipy.misc
import scipy.ndimage
import scipy.stats
from ..motion import blockMotion
from ..utils import *
def compute_niqe_features(frames):
    blocksizerow = 96
    blocksizecol = 96
    T, M, N, C = frames.shape
    assert (M >= blocksizerow * 2) & (N >= blocksizecol * 2), 'Video too small for NIQE extraction'
    module_path = dirname(__file__)
    params = scipy.io.loadmat(join(module_path, 'data', 'frames_modelparameters.mat'))
    mu_prisparam = params['mu_prisparam']
    cov_prisparam = params['cov_prisparam']
    niqe_features = np.zeros((frames.shape[0] - 10, 37))
    idx = 0
    for i in range(5, frames.shape[0] - 5):
        niqe_features[idx] = computequality(frames[i], blocksizerow, blocksizecol, mu_prisparam, cov_prisparam)
        idx += 1
    niqe_features = np.mean(niqe_features, axis=0)
    return niqe_features