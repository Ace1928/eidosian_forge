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
def binary_to_array(value, obj=None):
    return np.frombuffer(value['data'], dtype=value['dtype']).reshape(value['shape'])