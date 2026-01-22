from __future__ import annotations
import collections
import os
import sys
import warnings
import PIL
from . import Image
def get_supported_codecs():
    """
    :returns: A list of all supported codecs.
    """
    return [f for f in codecs if check_codec(f)]