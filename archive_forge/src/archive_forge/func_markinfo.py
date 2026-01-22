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
def markinfo(self) -> dict:
    """Return the PDF MarkInfo value."""
    xref = self.pdf_catalog()
    if xref == 0:
        return None
    rc = self.xref_get_key(xref, 'MarkInfo')
    if rc[0] == 'null':
        return {}
    if rc[0] == 'xref':
        xref = int(rc[1].split()[0])
        val = self.xref_object(xref, compressed=True)
    elif rc[0] == 'dict':
        val = rc[1]
    else:
        val = None
    if val is None or not (val[:2] == '<<' and val[-2:] == '>>'):
        return {}
    valid = {'Marked': False, 'UserProperties': False, 'Suspects': False}
    val = val[2:-2].split('/')
    for v in val[1:]:
        try:
            key, value = v.split()
        except Exception:
            if g_exceptions_verbose > 1:
                exception_info()
            return valid
        if value == 'true':
            valid[key] = True
    return valid