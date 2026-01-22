from __future__ import annotations
import collections
import os
import sys
import warnings
import PIL
from . import Image
def version_feature(feature):
    """
    :param feature: The feature to check for.
    :returns: The version number as a string, or ``None`` if not available.
    :raises ValueError: If the feature is not defined in this version of Pillow.
    """
    if not check_feature(feature):
        return None
    module, flag, ver = features[feature]
    if ver is None:
        return None
    return getattr(__import__(module, fromlist=[ver]), ver)