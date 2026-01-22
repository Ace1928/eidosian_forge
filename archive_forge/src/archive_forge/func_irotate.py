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
def irotate(rotation, initial=np.identity(3)):
    """Determine x, y, z rotation angles from rotation matrix."""
    a = np.dot(initial, rotation)
    cx, sx, rx = givens(a[2, 2], a[1, 2])
    cy, sy, ry = givens(rx, a[0, 2])
    cz, sz, rz = givens(cx * a[1, 1] - sx * a[2, 1], cy * a[0, 1] - sy * (sx * a[1, 1] + cx * a[2, 1]))
    x = degrees(atan2(sx, cx))
    y = degrees(atan2(-sy, cy))
    z = degrees(atan2(sz, cz))
    return (x, y, z)