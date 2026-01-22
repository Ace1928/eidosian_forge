import collections
import datetime
import functools
import os
import subprocess
import sys
import time
import errno
from contextlib import contextmanager
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import __version__ as PIL__version__
from PIL import ImageGrab
def pixelMatchesColor(x, y, expectedRGBColor, tolerance=0):
    """
    Return True if the pixel at x, y is matches the expected color of the RGB
    tuple, each color represented from 0 to 255, within an optional tolerance.
    """
    if isinstance(x, collections.abc.Sequence) and len(x) == 2:
        raise TypeError('pixelMatchesColor() has updated and no longer accepts a tuple of (x, y) values for the first argument. Pass these arguments as two separate arguments instead: pixelMatchesColor(x, y, rgb) instead of pixelMatchesColor((x, y), rgb)')
    pix = pixel(x, y)
    if len(pix) == 3 or len(expectedRGBColor) == 3:
        r, g, b = pix[:3]
        exR, exG, exB = expectedRGBColor[:3]
        return abs(r - exR) <= tolerance and abs(g - exG) <= tolerance and (abs(b - exB) <= tolerance)
    elif len(pix) == 4 and len(expectedRGBColor) == 4:
        r, g, b, a = pix
        exR, exG, exB, exA = expectedRGBColor
        return abs(r - exR) <= tolerance and abs(g - exG) <= tolerance and (abs(b - exB) <= tolerance) and (abs(a - exA) <= tolerance)
    else:
        assert False, 'Color mode was expected to be length 3 (RGB) or 4 (RGBA), but pixel is length %s and expectedRGBColor is length %s' % (len(pix), len(expectedRGBColor))