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
def create_array_binary_serialization(attrname, update_from_js=False):

    def from_json_to_array(value, obj=None):
        global last_value_to_array
        last_value_to_array = value
        if update_from_js:
            return np.array(value)
        else:
            return getattr(obj, attrname)
    return dict(to_json=array_to_binary_or_json, from_json=from_json_to_array)