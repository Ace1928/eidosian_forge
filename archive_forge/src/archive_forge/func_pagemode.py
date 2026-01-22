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
def pagemode(self) -> str:
    """Return the PDF PageMode value.
        """
    xref = self.pdf_catalog()
    if xref == 0:
        return None
    rc = self.xref_get_key(xref, 'PageMode')
    if rc[0] == 'null':
        return 'UseNone'
    if rc[0] == 'name':
        return rc[1][1:]
    return 'UseNone'