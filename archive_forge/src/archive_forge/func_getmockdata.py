from numpy.testing import assert_equal, assert_almost_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.motion
import skvideo.datasets
def getmockdata():
    frame1 = gauss(50, 50, 5, 100)
    frame2 = gauss(55, 50, 5, 100)
    videodata = []
    videodata.append(frame1)
    videodata.append(frame2)
    videodata = np.array(videodata)
    return videodata