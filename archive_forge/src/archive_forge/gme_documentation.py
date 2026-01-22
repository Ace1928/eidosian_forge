import numpy as np
import os
import scipy.ndimage
import scipy.spatial
from ..utils import *
Global motion estimation using edge features
    
    Given two frames, find a robust global translation vector
    found using edge information.

    Parameters
    ----------
    frame1 : ndarray
        first input frame, shape (1, M, N, C), (1, M, N), (M, N, C) or (M, N)

    frame2 : ndarray
        second input frame, shape (1, M, N, C), (1, M, N), (M, N, C) or (M, N)

    r : int
        Search radius for measuring correspondences.

    method : string
        "hamming" --> use Hamming distance when measuring edge correspondence distances. The distance used in the census transform. [#f1]_

        "hausdorff" --> use Hausdorff distance when measuring edge correspondence distances. [#f2]_

    Returns
    ----------
    globalMotionVector  : ndarray, shape (2,)
        The motion to minimize edge distances by moving frame2 with respect to frame1.

    References
    ----------
    .. [#f1] Ramin Zabih and John Woodfill. Non-parametric local transforms for computing visual correspondence. Computer Vision-ECCV, 151-158, 1994. 
    
    .. [#f2] Kevin Mai, Ramin Zabih, and Justin Miller. Feature-based algorithms for detecting and classifying scene breaks. Cornell University, 1995.

    