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
def get_lineart(self) -> object:
    """Get page drawings paths.

            Note:
            For greater comfort, this method converts point-like, rect-like, quad-like
            tuples of the C version to respective Point / Rect / Quad objects.
            Also adds default items that are missing in original path types.
            In contrast to get_drawings(), this output is an object.
            """
    val = self.get_cdrawings(extended=True)
    paths = self.Drawpathlist()
    for path in val:
        npath = self.Drawpath(**path)
        if npath.type != 'clip':
            npath.rect = Rect(path['rect'])
        else:
            npath.scissor = Rect(path['scissor'])
        if npath.type != 'group':
            items = path['items']
            newitems = []
            for item in items:
                cmd = item[0]
                rest = item[1:]
                if cmd == 're':
                    item = ('re', Rect(rest[0]).normalize(), rest[1])
                elif cmd == 'qu':
                    item = ('qu', Quad(rest[0]))
                else:
                    item = tuple([cmd] + [Point(i) for i in rest])
                newitems.append(item)
            npath.items = newitems
        if npath.type == 'f':
            npath.stroke_opacity = None
            npath.dashes = None
            npath.line_join = None
            npath.line_cap = None
            npath.color = None
            npath.width = None
        paths.append(npath)
    val = None
    return paths