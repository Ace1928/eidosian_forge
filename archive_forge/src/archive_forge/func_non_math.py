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
def non_math(self, toks: ParseResults) -> T.Any:
    s = toks[0].replace('\\$', '$')
    symbols = [Char(c, self.get_state()) for c in s]
    hlist = Hlist(symbols)
    self.push_state()
    self.get_state().font = mpl.rcParams['mathtext.default']
    return [hlist]