from __future__ import absolute_import
import cython
import os
import sys
import re
import io
import codecs
import glob
import shutil
import tempfile
from functools import wraps
from . import __version__ as cython_version
def long_literal(value):
    if isinstance(value, basestring):
        value = str_to_number(value)
    return not -2 ** 31 <= value < 2 ** 31