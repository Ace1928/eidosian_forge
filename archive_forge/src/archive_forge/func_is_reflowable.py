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
@property
def is_reflowable(self):
    """Check if document is layoutable."""
    if self.is_closed:
        raise ValueError('document closed')
    return mupdf.fz_is_document_reflowable(self._document())