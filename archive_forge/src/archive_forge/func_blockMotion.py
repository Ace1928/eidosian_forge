import numpy as np
import os
import time
from ..utils import *
def blockMotion(videodata, method='DS', mbSize=8, p=2, **plugin_args):
    """Block-based motion estimation
    
    Given a sequence of frames, this function
    returns motion vectors between frames.

    Parameters
    ----------
    videodata : ndarray, shape (numFrames, height, width, channel)
        A sequence of frames

    method : string
        "ES" --> exhaustive search

        "3SS" --> 3-step search

        "N3SS" --> "new" 3-step search [#f1]_

        "SE3SS" --> Simple and Efficient 3SS [#f2]_

        "4SS" --> 4-step search [#f3]_

        "ARPS" --> Adaptive Rood Pattern search [#f4]_

        "DS" --> Diamond search [#f5]_

    mbSize : int
        Macroblock size

    p : int
        Algorithm search distance parameter

    Returns
    ----------
    motionData : ndarray, shape (numFrames - 1, height/mbSize, width/mbSize, 2)

        The motion vectors computed from videodata. The first element of the last axis contains the y motion component, and second element contains the x motion component.

    References
    ----------
    .. [#f1] Renxiang Li, Bing Zeng, and Ming L. Liou, "A new three-step search algorithm for block motion estimation." IEEE Transactions on Circuits and Systems for Video Technology, 4 (4) 438-442, Aug 1994

    .. [#f2] Jianhua Lu and Ming L. Liou, "A simple and efficient search algorithm for block-matching motion estimation." IEEE Transactions on Circuits and Systems for Video Technology, 7 (2) 429-433, Apr 1997

    .. [#f3] Lai-Man Po and Wing-Chung Ma, "A novel four-step search algorithm for fast block motion estimation." IEEE Transactions on Circuits and Systems for Video Technology, 6 (3) 313-317, Jun 1996

    .. [#f4] Yao Nie and Kai-Kuang Ma, "Adaptive rood pattern search for fast block-matching motion estimation." IEEE Transactions on Image Processing, 11 (12) 1442-1448, Dec 2002

    .. [#f5] Shan Zhu and Kai-Kuang Ma, "A new diamond search algorithm for fast block-matching motion estimation." IEEE Transactions on Image Processing, 9 (2) 287-290, Feb 2000

    """
    videodata = vshape(videodata)
    luminancedata = rgb2gray(videodata)
    numFrames, height, width, channels = luminancedata.shape
    assert numFrames > 1, 'Must have more than 1 frame for motion estimation!'
    luminancedata = luminancedata.reshape((numFrames, height, width))
    motionData = np.zeros((numFrames - 1, np.int(height / mbSize), np.int(width / mbSize), 2), np.int8)
    if method == 'ES':
        for i in range(numFrames - 1):
            motion = _ES(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[i, :, :, :] = motion
    elif method == '4SS':
        for i in range(numFrames - 1):
            motion, comps = _4SS(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[i, :, :, :] = motion
    elif method == '3SS':
        for i in range(numFrames - 1):
            motion, comps = _3SS(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[i, :, :, :] = motion
    elif method == 'N3SS':
        for i in range(numFrames - 1):
            motion, comps = _N3SS(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[i, :, :, :] = motion
    elif method == 'SE3SS':
        for i in range(numFrames - 1):
            motion, comps = _SE3SS(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[i, :, :, :] = motion
    elif method == 'ARPS':
        for i in range(numFrames - 1):
            motion, comps = _ARPS(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[i, :, :, :] = motion
    elif method == 'DS':
        for i in range(numFrames - 1):
            motion, comps = _DS(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
            motionData[i, :, :, :] = motion
    else:
        raise NotImplementedError
    return motionData