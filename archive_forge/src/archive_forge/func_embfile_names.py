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
def embfile_names(self) -> list:
    """Get list of names of EmbeddedFiles."""
    filenames = []
    self._embfile_names(filenames)
    return filenames