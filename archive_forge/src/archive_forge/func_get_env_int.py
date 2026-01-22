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
def get_env_int(name, default):
    """
    Returns `True`, `False` or `default` depending on whether $<name> is '1',
    '0' or unset. Otherwise assert-fails.
    """
    v = os.environ.get(name)
    if v is None:
        ret = default
    else:
        ret = int(v)
    if ret != default:
        log(f'Using non-default setting from {name}: {v}')
    return ret