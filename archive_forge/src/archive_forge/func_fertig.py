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
def fertig(font):
    if not font.m_internal:
        raise RuntimeError(MSG_FONT_FAILED)
    if not font.m_internal.flags.never_embed:
        mupdf.fz_set_font_embedding(font, embed)
    return font