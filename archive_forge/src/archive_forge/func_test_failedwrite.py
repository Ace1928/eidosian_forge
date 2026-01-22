import skvideo.io
import sys
import numpy as np
import hashlib
import os
from numpy.testing import assert_equal
from nose.tools import *
@raises(AssertionError)
def test_failedwrite():
    np.random.seed(0)
    outputdata = np.random.random(size=(5, 480, 640, 3)) * 255
    outputdata = outputdata.astype(np.uint8)
    skvideo.io.vwrite('garbage/garbage.mp4', outputdata)