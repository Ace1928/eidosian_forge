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
def set_pagemode(self, pagemode: str):
    """Set the PDF PageMode value."""
    valid = ('UseNone', 'UseOutlines', 'UseThumbs', 'FullScreen', 'UseOC', 'UseAttachments')
    xref = self.pdf_catalog()
    if xref == 0:
        raise ValueError('not a PDF')
    if not pagemode:
        raise ValueError('bad PageMode value')
    if pagemode[0] == '/':
        pagemode = pagemode[1:]
    for v in valid:
        if pagemode.lower() == v.lower():
            self.xref_set_key(xref, 'PageMode', f'/{v}')
            return True
    raise ValueError('bad PageMode value')