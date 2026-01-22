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
def showthis(msg, cat, filename, lineno, file=None, line=None):
    text = warnings.formatwarning(msg, cat, filename, lineno, line=line)
    s = text.find('FitzDeprecation')
    if s < 0:
        log(text, file=sys.stderr)
        return
    text = text[s:].splitlines()[0][4:]
    log(text, file=sys.stderr)