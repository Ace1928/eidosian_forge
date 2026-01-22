from __future__ import division
import logging
import warnings
import math
from base64 import b64encode
import numpy as np
import PIL.Image
import ipywidgets
import ipywebrtc
from ipython_genutils.py3compat import string_types
from ipyvolume import utils
def flatten_frames(image):
    frames = []
    index = 0
    while True:
        try:
            image.seek(index)
        except EOFError:
            break
        frames.append(image.copy())
        index += 1
    return frames