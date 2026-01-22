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
def extractSelection(self, pointa, pointb):
    a = JM_point_from_py(pointa)
    b = JM_point_from_py(pointb)
    found = mupdf.fz_copy_selection(self.this, a, b, 0)
    return found