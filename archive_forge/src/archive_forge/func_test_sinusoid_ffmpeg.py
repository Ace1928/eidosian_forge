import skvideo.io
import skvideo.utils
import numpy as np
import os
import sys
@unittest.skipIf(not skvideo._HAS_FFMPEG, 'FFmpeg required for this test.')
def test_sinusoid_ffmpeg():
    pattern_sinusoid('ffmpeg')