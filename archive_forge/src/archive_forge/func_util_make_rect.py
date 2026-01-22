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
def util_make_rect(*args, p0=None, p1=None, x0=None, y0=None, x1=None, y1=None):
    """
    Helper for initialising rectangle classes.
    
    2022-09-02: This is quite different from PyMuPDF's util_make_rect(), which
    uses `goto` in ways that don't easily translate to Python.

    Returns (x0, y0, x1, y1) derived from <args>, then override with p0, p1,
    x0, y0, x1, y1 if they are not None.

    Accepts following forms for <args>:
        () returns all zeros.
        (top-left, bottom-right)
        (top-left, x1, y1)
        (x0, y0, bottom-right)
        (x0, y0, x1, y1)
        (rect)

    Where top-left and bottom-right are (x, y) or something with .x, .y
    members; rect is something with .x0, .y0, .x1, and .y1 members.

    2023-11-18: we now override with p0, p1, x0, y0, x1, y1 if not None.
    """

    def get_xy(arg):
        if isinstance(arg, (list, tuple)) and len(arg) == 2:
            return (arg[0], arg[1])
        if isinstance(arg, (Point, mupdf.FzPoint, mupdf.fz_point)):
            return (arg.x, arg.y)
        return (None, None)

    def make_tuple(a):
        if isinstance(a, tuple):
            return a
        if isinstance(a, Point):
            return (a.x, a.y)
        elif isinstance(a, (Rect, IRect, mupdf.FzRect, mupdf.fz_rect)):
            return (a.x0, a.y0, a.x1, a.y1)
        if not isinstance(a, (list, tuple)):
            a = (a,)
        return a

    def handle_args():
        if len(args) == 0:
            return (0, 0, 0, 0)
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, (list, tuple)) and len(arg) == 2:
                p1, p2 = arg
                return (*p1, *p2)
            if isinstance(arg, (list, tuple)) and len(arg) == 3:
                a, b, c = arg
                a = make_tuple(a)
                b = make_tuple(b)
                c = make_tuple(c)
                ret = (*a, *b, *c)
                return ret
            arg = make_tuple(arg)
            return arg
        elif len(args) == 2:
            return get_xy(args[0]) + get_xy(args[1])
        elif len(args) == 3:
            x0, y0 = get_xy(args[0])
            if (x0, y0) != (None, None):
                return (x0, y0, args[1], args[2])
            x1, y1 = get_xy(args[2])
            if (x1, y1) != (None, None):
                return (args[0], args[1], x1, y1)
        elif len(args) == 4:
            return (args[0], args[1], args[2], args[3])
        raise Exception(f'Unrecognised args: {args}')
    ret_x0, ret_y0, ret_x1, ret_y1 = handle_args()
    if p0 is not None:
        ret_x0, ret_y0 = get_xy(p0)
    if p1 is not None:
        ret_x1, ret_y1 = get_xy(p1)
    if x0 is not None:
        ret_x0 = x0
    if y0 is not None:
        ret_y0 = y0
    if x1 is not None:
        ret_x1 = x1
    if y1 is not None:
        ret_y1 = y1
    return (ret_x0, ret_y0, ret_x1, ret_y1)