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
def detect_super_script(line, ch):
    if line.m_internal.wmode == 0 and line.m_internal.dir.x == 1 and (line.m_internal.dir.y == 0):
        return ch.m_internal.origin.y < line.m_internal.first_char.origin.y - ch.m_internal.size * 0.1
    return 0