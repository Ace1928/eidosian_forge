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
@staticmethod
def color_text(color):
    if type(color) is str:
        return color
    if type(color) is int:
        return f'rgb({sRGB_to_rgb(color)})'
    if type(color) in (tuple, list):
        return f'rgb{tuple(color)}'
    return color