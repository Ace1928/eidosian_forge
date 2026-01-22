import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def prerotate(self, theta):
    """Calculate pre rotation and replace current matrix."""
    theta = float(theta)
    while theta < 0:
        theta += 360
    while theta >= 360:
        theta -= 360
    if abs(0 - theta) < EPSILON:
        pass
    elif abs(90.0 - theta) < EPSILON:
        a = self.a
        b = self.b
        self.a = self.c
        self.b = self.d
        self.c = -a
        self.d = -b
    elif abs(180.0 - theta) < EPSILON:
        self.a = -self.a
        self.b = -self.b
        self.c = -self.c
        self.d = -self.d
    elif abs(270.0 - theta) < EPSILON:
        a = self.a
        b = self.b
        self.a = -self.c
        self.b = -self.d
        self.c = a
        self.d = b
    else:
        rad = math.radians(theta)
        s = math.sin(rad)
        c = math.cos(rad)
        a = self.a
        b = self.b
        self.a = c * a + s * self.c
        self.b = c * b + s * self.d
        self.c = -s * a + c * self.c
        self.d = -s * b + c * self.d
    return self