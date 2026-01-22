import os
import sys
import numpy as np
from numpy.testing import assert_equal
import skvideo.datasets
import skvideo.io
@unittest.skipIf(not skvideo._HAS_FFMPEG, 'FFmpeg required for this test.')
def test_FFmpegWriter():
    outputfile = sys._getframe().f_code.co_name + '.mp4'
    outputdata = np.random.random(size=(5, 480, 640, 3)) * 255
    outputdata = outputdata.astype(np.uint8)
    writer = skvideo.io.FFmpegWriter(outputfile)
    for i in range(5):
        writer.writeFrame(outputdata[i])
    writer.close()
    os.remove(outputfile)