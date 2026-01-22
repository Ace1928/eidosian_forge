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
def get_textbox(page: Page, rect: rect_like, textpage=None) -> str:
    tp = textpage
    if tp is None:
        tp = page.get_textpage()
    elif getattr(tp, 'parent') != page:
        raise ValueError('not a textpage of this page')
    rc = tp.extractTextbox(rect)
    if textpage is None:
        del tp
    return rc