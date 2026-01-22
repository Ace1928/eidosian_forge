import errno
import functools
import os
import io
import pickle
import sys
import time
import string
import warnings
from importlib import import_module
from math import sin, cos, radians, atan2, degrees
from contextlib import contextmanager, ExitStack
from math import gcd
from pathlib import PurePath, Path
import re
import numpy as np
from ase.formula import formula_hill, formula_metal
def hsv2rgb(h, s, v):
    """http://en.wikipedia.org/wiki/HSL_and_HSV

    h (hue) in [0, 360[
    s (saturation) in [0, 1]
    v (value) in [0, 1]

    return rgb in range [0, 1]
    """
    if v == 0:
        return (0, 0, 0)
    if s == 0:
        return (v, v, v)
    i, f = divmod(h / 60.0, 1)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    if i == 0:
        return (v, t, p)
    elif i == 1:
        return (q, v, p)
    elif i == 2:
        return (p, v, t)
    elif i == 3:
        return (p, q, v)
    elif i == 4:
        return (t, p, v)
    elif i == 5:
        return (v, p, q)
    else:
        raise RuntimeError('h must be in [0, 360]')