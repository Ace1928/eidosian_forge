from ..utils import *
from .ssim import *
import numpy as np
import scipy.ndimage
def msssim(referenceVideoData, distortedVideoData, method='product'):
    """Computes Multiscale Structural Similarity (MS-SSIM) Index. [#f1]_

    Both video inputs are compared frame-by-frame to obtain T
    MS-SSIM measurements on the luminance channel.

    Parameters
    ----------
    referenceVideoData : ndarray
        Reference video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels. Here C is only allowed to be 1.

    distortedVideoData : ndarray
        Distorted video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels. Here C is only allowed to be 1.

    method : str
        Whether to use "product" (default) or to use "sum" for combing multiple scales into the single score.

    Returns
    -------
    msssim_array : ndarray
        The MS-SSIM results, ndarray of dimension (T,), where T
        is the number of frames

    References
    ----------

    .. [#f1] Z. Wang, E. P. Simoncelli and A. C. Bovik, "Multi-scale structural similarity for image quality assessment," IEEE Asilomar Conference Signals, Systems and Computers, Nov. 2003.

    """
    referenceVideoData = vshape(referenceVideoData)
    distortedVideoData = vshape(distortedVideoData)
    assert referenceVideoData.shape == distortedVideoData.shape
    T, M, N, C = referenceVideoData.shape
    assert C == 1, 'MS-SSIM called with videos containing %d channels. Please supply only the luminance channel' % (C,)
    assert (M >= 176) & (N >= 176), 'You supplied a resolution of %dx%d. MS-SSIM can only be used with videos large enough having multiple scales. Please use only with resolutions >= 176x176.' % (M, N)
    scores = np.zeros(T, dtype=np.float32)
    for t in range(T):
        referenceFrame = referenceVideoData[t, :, :, 0].astype(np.float32)
        distortedFrame = distortedVideoData[t, :, :, 0].astype(np.float32)
        scores[t] = compute_msssim(referenceFrame, distortedFrame, method=method)
    return scores