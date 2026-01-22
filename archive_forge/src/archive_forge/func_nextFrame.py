import os
import time
import warnings
import numpy as np
from .. import _HAS_FFMPEG
from ..utils import *
def nextFrame(self):
    """Yields frames using a generator

        Returns T ndarrays of size (M, N, C), where T is number of frames,
        M is height, N is width, and C is number of channels per pixel.

        """
    if self.inputframenum == 0:
        while True:
            frame = self._readFrame()
            if len(frame) == 0:
                break
            yield frame
    else:
        for i in range(self.inputframenum):
            frame = self._readFrame()
            if len(frame) == 0:
                break
            yield frame