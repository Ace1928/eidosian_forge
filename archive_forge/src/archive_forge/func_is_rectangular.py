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
@property
def is_rectangular(self):
    """Check if quad is rectangular.

        Notes:
            Some rotation matrix can thus transform it into a rectangle.
            This is equivalent to three corners enclose 90 degrees.
        Returns:
            True or False.
        """
    sine = util_sine_between(self.ul, self.ur, self.lr)
    if abs(sine - 1) > EPSILON:
        return False
    sine = util_sine_between(self.ur, self.lr, self.ll)
    if abs(sine - 1) > EPSILON:
        return False
    sine = util_sine_between(self.lr, self.ll, self.ul)
    if abs(sine - 1) > EPSILON:
        return False
    return True