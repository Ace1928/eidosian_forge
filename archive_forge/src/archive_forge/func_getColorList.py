import io
import math
import os
import typing
import weakref
def getColorList() -> list:
    """
    Returns a list of just the colour names used by this module.
    :rtype: list of strings
    """
    return [x[0] for x in getColorInfoList()]