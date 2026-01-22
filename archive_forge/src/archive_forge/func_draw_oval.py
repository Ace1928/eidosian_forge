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
def draw_oval(self, tetra: typing.Union[quad_like, rect_like]):
    """Draw an ellipse inside a tetrapod."""
    if len(tetra) != 4:
        raise ValueError('invalid arg length')
    if hasattr(tetra[0], '__float__'):
        q = Rect(tetra).quad
    else:
        q = Quad(tetra)
    mt = q.ul + (q.ur - q.ul) * 0.5
    mr = q.ur + (q.lr - q.ur) * 0.5
    mb = q.ll + (q.lr - q.ll) * 0.5
    ml = q.ul + (q.ll - q.ul) * 0.5
    if not self.last_point == ml:
        self.draw_cont += '%g %g m\n' % JM_TUPLE(ml * self.ipctm)
        self.last_point = ml
    self.draw_curve(ml, q.ll, mb)
    self.draw_curve(mb, q.lr, mr)
    self.draw_curve(mr, q.ur, mt)
    self.draw_curve(mt, q.ul, ml)
    self.updateRect(q.rect)
    self.last_point = ml
    return self.last_point