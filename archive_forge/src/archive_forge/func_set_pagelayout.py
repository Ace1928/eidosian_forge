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
def set_pagelayout(self, pagelayout: str):
    """Set the PDF PageLayout value."""
    valid = ('SinglePage', 'OneColumn', 'TwoColumnLeft', 'TwoColumnRight', 'TwoPageLeft', 'TwoPageRight')
    xref = self.pdf_catalog()
    if xref == 0:
        raise ValueError('not a PDF')
    if not pagelayout:
        raise ValueError('bad PageLayout value')
    if pagelayout[0] == '/':
        pagelayout = pagelayout[1:]
    for v in valid:
        if pagelayout.lower() == v.lower():
            self.xref_set_key(xref, 'PageLayout', f'/{v}')
            return True
    raise ValueError('bad PageLayout value')