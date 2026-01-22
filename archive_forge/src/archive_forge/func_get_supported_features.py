from __future__ import annotations
import collections
import os
import sys
import warnings
import PIL
from . import Image
def get_supported_features():
    """
    :returns: A list of all supported features.
    """
    return [f for f in features if check_feature(f)]