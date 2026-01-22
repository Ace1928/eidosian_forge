from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
def onScreen(x, y=None):
    """Returns whether the given xy coordinates are on the primary screen or not.

    Note that this function doesn't work for secondary screens.

    Args:
      Either the arguments are two separate values, first arg for x and second
        for y, or there is a single argument of a sequence with two values, the
        first x and the second y.
        Example: onScreen(x, y) or onScreen([x, y])

    Returns:
      bool: True if the xy coordinates are on the screen at its current
        resolution, otherwise False.
    """
    x, y = _normalizeXYArgs(x, y)
    x = int(x)
    y = int(y)
    width, height = platformModule._size()
    return 0 <= x < width and 0 <= y < height