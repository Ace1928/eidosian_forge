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
def find_string(s, needle):
    assert isinstance(s, str)
    for i in range(len(s)):
        end = match_string(s[i:], needle)
        if end is not None:
            end += i
            return (i, end)
    return (None, None)