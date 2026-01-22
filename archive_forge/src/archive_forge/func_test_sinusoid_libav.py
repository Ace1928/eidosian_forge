import skvideo.io
import skvideo.utils
import numpy as np
import os
import sys
@unittest.skipIf(not skvideo._HAS_AVCONV, 'LibAV required for this test.')
def test_sinusoid_libav():
    pattern_sinusoid('libav')