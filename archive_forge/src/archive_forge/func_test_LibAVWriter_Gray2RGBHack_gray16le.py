import os
import sys
import numpy as np
from numpy.testing import assert_equal, assert_array_less
import skvideo
import skvideo.datasets
import skvideo.io
@unittest.skipIf(not skvideo._HAS_AVCONV, 'LibAV required for this test.')
def test_LibAVWriter_Gray2RGBHack_gray16le():
    _Gray2RGBHack_Helper('gray16le')