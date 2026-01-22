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
def jm_lineart_clip_text(dev, ctx, text, ctm, scissor):
    if not dev.clips:
        return
    compute_scissor(dev)
    dev.depth += 1