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
def pdfobj_string(o, prefix=''):
    """
    Returns description of mupdf.PdfObj (wrapper for pdf_obj) <o>.
    """
    assert 0, 'use mupdf.pdf_debug_obj() ?'
    ret = ''
    if mupdf.pdf_is_array(o):
        l = mupdf.pdf_array_len(o)
        ret += f'array {l}\n'
        for i in range(l):
            oo = mupdf.pdf_array_get(o, i)
            ret += pdfobj_string(oo, prefix + '    ')
            ret += '\n'
    elif mupdf.pdf_is_bool(o):
        ret += f'bool: {o.array_get_bool()}\n'
    elif mupdf.pdf_is_dict(o):
        l = mupdf.pdf_dict_len(o)
        ret += f'dict {l}\n'
        for i in range(l):
            key = mupdf.pdf_dict_get_key(o, i)
            value = mupdf.pdf_dict_get(o, key)
            ret += f'{prefix} {key}: '
            ret += pdfobj_string(value, prefix + '    ')
            ret += '\n'
    elif mupdf.pdf_is_embedded_file(o):
        ret += f'embedded_file: {o.embedded_file_name()}\n'
    elif mupdf.pdf_is_indirect(o):
        ret += f'indirect: ...\n'
    elif mupdf.pdf_is_int(o):
        ret += f'int: {mupdf.pdf_to_int(o)}\n'
    elif mupdf.pdf_is_jpx_image(o):
        ret += f'jpx_image:\n'
    elif mupdf.pdf_is_name(o):
        ret += f'name: {mupdf.pdf_to_name(o)}\n'
    elif o.pdf_is_null:
        ret += f'null\n'
    elif o.pdf_is_real:
        ret += f'real: {o.pdf_to_real()}\n'
    elif mupdf.pdf_is_stream(o):
        ret += f'stream\n'
    elif mupdf.pdf_is_string(o):
        ret += f'string: {mupdf.pdf_to_string(o)}\n'
    else:
        ret += '<>\n'
    return ret