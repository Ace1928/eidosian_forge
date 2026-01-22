from __future__ import annotations
import collections
import os
import sys
import warnings
import PIL
from . import Image
def check_codec(feature):
    """
    Checks if a codec is available.

    :param feature: The codec to check for.
    :returns: ``True`` if available, ``False`` otherwise.
    :raises ValueError: If the codec is not defined in this version of Pillow.
    """
    if feature not in codecs:
        msg = f'Unknown codec {feature}'
        raise ValueError(msg)
    codec, lib = codecs[feature]
    return codec + '_encoder' in dir(Image.core)