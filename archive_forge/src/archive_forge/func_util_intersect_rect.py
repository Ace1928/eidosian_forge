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
def util_intersect_rect(r1, r2):
    return JM_py_from_rect(mupdf.fz_intersect_rect(JM_rect_from_py(r1), JM_rect_from_py(r2)))