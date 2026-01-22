import os
import subprocess as sp
import sys
from moviepy.config import get_setting
from moviepy.tools import subprocess_call
def ffmpeg_resize(video, output, size):
    """ resizes ``video`` to new size ``size`` and write the result
        in file ``output``. """
    cmd = [get_setting('FFMPEG_BINARY'), '-i', video, '-vf', 'scale=%d:%d' % (size[0], size[1]), output]
    subprocess_call(cmd)