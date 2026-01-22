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
def obj_string(obj):
    """Return string version of a PDF object definition."""
    buffer = mupdf.fz_new_buffer(512)
    output = mupdf.FzOutput(buffer)
    mupdf.pdf_print_obj(output, obj, 1, 0)
    output.fz_close_output()
    return JM_UnicodeFromBuffer(buffer)