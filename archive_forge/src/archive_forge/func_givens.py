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
def givens(a, b):
    """Solve the equation system::

      [ c s]   [a]   [r]
      [    ] . [ ] = [ ]
      [-s c]   [b]   [0]
    """
    sgn = np.sign
    if b == 0:
        c = sgn(a)
        s = 0
        r = abs(a)
    elif abs(b) >= abs(a):
        cot = a / b
        u = sgn(b) * (1 + cot ** 2) ** 0.5
        s = 1.0 / u
        c = s * cot
        r = b * u
    else:
        tan = b / a
        u = sgn(a) * (1 + tan ** 2) ** 0.5
        c = 1.0 / u
        s = c * tan
        r = a * u
    return (c, s, r)