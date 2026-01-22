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
def set_bold(self, val=True):
    """Set bold on / off via CSS style"""
    if val:
        val = 'bold'
    else:
        val = 'normal'
    text = 'font-weight: %s' % val
    self.append_styled_span(text)
    return self