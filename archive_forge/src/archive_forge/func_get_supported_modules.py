from __future__ import annotations
import collections
import os
import sys
import warnings
import PIL
from . import Image
def get_supported_modules():
    """
    :returns: A list of all supported modules.
    """
    return [f for f in modules if check_module(f)]