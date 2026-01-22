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
def compute_scissor(dev):
    """
    Every scissor of a clip is a sub rectangle of the preceeding clip scissor
    if the clip level is larger.
    """
    if dev.scissors is None:
        dev.scissors = list()
    num_scissors = len(dev.scissors)
    if num_scissors > 0:
        last_scissor = dev.scissors[num_scissors - 1]
        scissor = JM_rect_from_py(last_scissor)
        scissor = mupdf.fz_intersect_rect(scissor, dev.pathrect)
    else:
        scissor = dev.pathrect
    dev.scissors.append(JM_py_from_rect(scissor))
    return scissor