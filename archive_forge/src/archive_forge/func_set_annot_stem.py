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
@staticmethod
def set_annot_stem(stem=None):
    global JM_annot_id_stem
    if stem is None:
        return JM_annot_id_stem
    len_ = len(stem) + 1
    if len_ > 50:
        len_ = 50
    JM_annot_id_stem = stem[:50]
    return JM_annot_id_stem