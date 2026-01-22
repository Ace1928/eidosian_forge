import os
import numpy as np
import proglog
from tqdm import tqdm
from moviepy.audio.io.ffmpeg_audiowriter import ffmpeg_audiowrite
from moviepy.Clip import Clip
from moviepy.decorators import requires_duration
from moviepy.tools import deprecated_version_of, extensions_dict
def max_volume(self, stereo=False, chunksize=50000, logger=None):
    stereo = stereo and self.nchannels == 2
    maxi = np.array([0, 0]) if stereo else 0
    for chunk in self.iter_chunks(chunksize=chunksize, logger=logger):
        maxi = np.maximum(maxi, abs(chunk).max(axis=0)) if stereo else max(maxi, abs(chunk).max())
    return maxi