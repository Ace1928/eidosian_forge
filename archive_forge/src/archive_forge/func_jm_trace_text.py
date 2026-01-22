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
def jm_trace_text(dev, text, type_, ctm, colorspace, color, alpha, seqno):
    span = text.head
    while 1:
        if not span:
            break
        jm_trace_text_span(dev, span, type_, ctm, colorspace, color, alpha, seqno)
        span = span.next