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
def include_rect(self, r):
    """Extend rectangle to include rectangle r."""
    rect = self.rect.include_rect(r)
    return rect.irect