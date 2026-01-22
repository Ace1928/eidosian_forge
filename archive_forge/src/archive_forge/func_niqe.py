from ..utils import *
import numpy as np
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
def niqe(inputVideoData):
    """Computes Naturalness Image Quality Evaluator. [#f1]_

    Input a video of any quality and get back its distance frame-by-frame
    from naturalness.

    Parameters
    ----------
    inputVideoData : ndarray
        Input video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels. Here C is only allowed to be 1.

    Returns
    -------
    niqe_array : ndarray
        The niqe results, ndarray of dimension (T,), where T
        is the number of frames

    References
    ----------

    .. [#f1] Mittal, Anish, Rajiv Soundararajan, and Alan C. Bovik. "Making a 'completely blind' image quality analyzer." IEEE Signal Processing Letters 20.3 (2013): 209-212.

    """
    patch_size = 96
    module_path = dirname(__file__)
    params = scipy.io.loadmat(join(module_path, 'data', 'niqe_image_params.mat'))
    pop_mu = np.ravel(params['pop_mu'])
    pop_cov = params['pop_cov']
    inputVideoData = vshape(inputVideoData)
    T, M, N, C = inputVideoData.shape
    assert C == 1, 'niqe called with videos containing %d channels. Please supply only the luminance channel' % (C,)
    assert M > patch_size * 2 + 1, 'niqe called with small frame size, requires > 192x192 resolution video using current training parameters'
    assert N > patch_size * 2 + 1, 'niqe called with small frame size, requires > 192x192 resolution video using current training parameters'
    niqe_scores = np.zeros(T, dtype=np.float32)
    for t in range(T):
        feats = get_patches_test_features(inputVideoData[t, :, :, 0], patch_size)
        sample_mu = np.mean(feats, axis=0)
        sample_cov = np.cov(feats.T)
        X = sample_mu - pop_mu
        covmat = (pop_cov + sample_cov) / 2.0
        pinvmat = scipy.linalg.pinv(covmat)
        niqe_scores[t] = np.sqrt(np.dot(np.dot(X, pinvmat), X))
    return niqe_scores