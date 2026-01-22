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
def set_markinfo(self, markinfo: dict) -> bool:
    """Set the PDF MarkInfo values."""
    xref = self.pdf_catalog()
    if xref == 0:
        raise ValueError('not a PDF')
    if not markinfo or not isinstance(markinfo, dict):
        return False
    valid = {'Marked': False, 'UserProperties': False, 'Suspects': False}
    if not set(valid.keys()).issuperset(markinfo.keys()):
        badkeys = f'bad MarkInfo key(s): {set(markinfo.keys()).difference(valid.keys())}'
        raise ValueError(badkeys)
    pdfdict = '<<'
    valid.update(markinfo)
    for key, value in valid.items():
        value = str(value).lower()
        if value not in ('true', 'false'):
            raise ValueError(f"bad key value '{key}': '{value}'")
        pdfdict += f'/{key} {value}'
    pdfdict += '>>'
    self.xref_set_key(xref, 'MarkInfo', pdfdict)
    return True