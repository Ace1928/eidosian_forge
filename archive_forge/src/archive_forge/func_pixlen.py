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
def pixlen(x):
    """Calculate pixel length of x."""
    if ordering < 0:
        return sum([glyphs[ord(c)][1] for c in x]) * fontsize
    else:
        return len(x) * fontsize