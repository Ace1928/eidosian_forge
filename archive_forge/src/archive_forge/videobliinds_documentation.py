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
Computes Video Bliinds features. [#f1]_

    Since this is a referenceless quality algorithm, only 1 video is needed. This function
    provides the raw features used by the algorithm.

    Parameters
    ----------
    videoData : ndarray
        Reference video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels.

    Returns
    -------
    features : ndarray, shape (46,)
        |  The individual features of the algorithm. The features are arranged as follows:
        |  
        |    features[:36] : spatial niqe vector averaged over the video, shape (36,)
        |    features[36] : niqe naturalness score, shape (1,)
        |    features[37:39] : DC measurements between frames, shape (2,)
        |    features[39:44] : Natural Video Statistics, shape (5,)
        |    features[44] : Motion coherence, shape (1,)
        |    features[45] : Global motion, shape (1,)

    References
    ----------

    .. [#f1] M. Saad and A.C. Bovik, "Blind prediction of natural video quality" IEEE Transactions on Image Processing, December 2013.

    