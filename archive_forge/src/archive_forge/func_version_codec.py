from __future__ import annotations
import collections
import os
import sys
import warnings
import PIL
from . import Image
def version_codec(feature):
    """
    :param feature: The codec to check for.
    :returns:
        The version number as a string, or ``None`` if not available.
        Checked at compile time for ``jpg``, run-time otherwise.
    :raises ValueError: If the codec is not defined in this version of Pillow.
    """
    if not check_codec(feature):
        return None
    codec, lib = codecs[feature]
    version = getattr(Image.core, lib + '_version')
    if feature == 'libtiff':
        return version.split('\n')[0].split('Version ')[1]
    return version