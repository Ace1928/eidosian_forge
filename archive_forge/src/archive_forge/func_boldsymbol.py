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
def boldsymbol(self, toks: ParseResults) -> T.Any:
    self.push_state()
    state = self.get_state()
    hlist: list[Node] = []
    name = toks['value']
    for c in name:
        if isinstance(c, Hlist):
            k = c.children[1]
            if isinstance(k, Char):
                k.font = 'bf'
                k._update_metrics()
            hlist.append(c)
        elif isinstance(c, Char):
            c.font = 'bf'
            if c.c in self._latin_alphabets or c.c[1:] in self._small_greek:
                c.font = 'bfit'
                c._update_metrics()
            c._update_metrics()
            hlist.append(c)
        else:
            hlist.append(c)
    self.pop_state()
    return Hlist(hlist)