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
def begin_page(self, mediabox):
    mediabox2 = JM_rect_from_py(mediabox)
    device = mupdf.fz_begin_page(self.this, mediabox2)
    device_wrapper = DeviceWrapper(device)
    return device_wrapper