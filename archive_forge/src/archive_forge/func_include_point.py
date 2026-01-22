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
def include_point(self, p):
    """Extend rectangle to include point p."""
    rect = self.rect.include_point(p)
    return rect.irect