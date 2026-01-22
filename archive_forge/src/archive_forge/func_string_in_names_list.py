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
def string_in_names_list(p, names_list):
    n = mupdf.pdf_array_len(names_list) if names_list else 0
    str_ = mupdf.pdf_to_text_string(p)
    for i in range(0, n, 2):
        if mupdf.pdf_to_text_string(mupdf.pdf_array_get(names_list, i)) == str_:
            return 1
    return 0