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
def util_round_rect(rect):
    return JM_py_from_irect(mupdf.fz_round_rect(JM_rect_from_py(rect)))