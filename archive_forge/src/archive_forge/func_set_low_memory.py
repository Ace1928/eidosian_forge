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
def set_low_memory(on=None):
    """Set / unset MuPDF device caching."""
    global g_no_device_caching
    if on is not None:
        g_no_device_caching = bool(on)
    return g_no_device_caching