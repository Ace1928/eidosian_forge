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
def span_bottom(self):
    """Find deepest level in stacked spans."""
    parent = self
    child = self.last_child
    if child is None:
        return None
    while child.is_text:
        child = child.previous
        if child is None:
            break
    if child is None or child.tagname != 'span':
        return None
    while True:
        if child is None:
            return parent
        if child.tagname in ('a', 'sub', 'sup', 'body') or child.is_text:
            child = child.next
            continue
        if child.tagname == 'span':
            parent = child
            child = child.first_child
        else:
            return parent