import os
import subprocess as sp
import warnings
import numpy as np
from moviepy.compat import DEVNULL, PY3
from moviepy.config import get_setting
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
def skip_chunk(self, chunksize):
    s = self.proc.stdout.read(self.nchannels * chunksize * self.nbytes)
    self.proc.stdout.flush()
    self.pos = self.pos + chunksize