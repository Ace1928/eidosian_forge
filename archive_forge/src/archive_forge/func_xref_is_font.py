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
def xref_is_font(self, xref):
    """Check if xref is a font object."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    if self.xref_get_key(xref, 'Type')[1] == '/Font':
        return True
    return False