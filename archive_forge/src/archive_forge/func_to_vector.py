from __future__ import annotations
import abc
import copy
import enum
import functools
import logging
import os
import re
import types
import unicodedata
import string
import typing as T
from typing import NamedTuple
import numpy as np
from pyparsing import (
import matplotlib as mpl
from . import cbook
from ._mathtext_data import (
from .font_manager import FontProperties, findfont, get_font
from .ft2font import FT2Font, FT2Image, KERNING_DEFAULT
from packaging.version import parse as parse_version
from pyparsing import __version__ as pyparsing_version
def to_vector(self) -> VectorParse:
    w, h, d = map(np.ceil, [self.box.width, self.box.height, self.box.depth])
    gs = [(info.font, info.fontsize, info.num, ox, h - oy + info.offset) for ox, oy, info in self.glyphs]
    rs = [(x1, h - y2, x2 - x1, y2 - y1) for x1, y1, x2, y2 in self.rects]
    return VectorParse(w, h + d, d, gs, rs)