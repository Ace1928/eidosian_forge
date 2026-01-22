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
def set_word_spacing(self, spacing):
    """Set inter-word spacing value via CSS style"""
    text = f'word-spacing: {spacing}'
    self.append_styled_span(text)
    return self