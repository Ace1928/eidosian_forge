import os
import subprocess as sp
import warnings
import numpy as np
from moviepy.compat import DEVNULL, PY3
from moviepy.config import get_setting
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
def buffer_around(self, framenumber):
    """
        Fills the buffer with frames, centered on ``framenumber``
        if possible
        """
    new_bufferstart = max(0, framenumber - self.buffersize // 2)
    if self.buffer is not None:
        current_f_end = self.buffer_startframe + self.buffersize
        if new_bufferstart < current_f_end < new_bufferstart + self.buffersize:
            conserved = current_f_end - new_bufferstart + 1
            chunksize = self.buffersize - conserved
            array = self.read_chunk(chunksize)
            self.buffer = np.vstack([self.buffer[-conserved:], array])
        else:
            self.seek(new_bufferstart)
            self.buffer = self.read_chunk(self.buffersize)
    else:
        self.seek(new_bufferstart)
        self.buffer = self.read_chunk(self.buffersize)
    self.buffer_startframe = new_bufferstart