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
@functools.cache
def get_sized_alternatives_for_symbol(self, fontname: str, sym: str) -> list[tuple[str, str]] | list[tuple[int, str]]:
    fixes = {'\\{': '{', '\\}': '}', '\\[': '[', '\\]': ']', '<': '⟨', '>': '⟩'}
    sym = fixes.get(sym, sym)
    try:
        uniindex = get_unicode_index(sym)
    except ValueError:
        return [(fontname, sym)]
    alternatives = [(i, chr(uniindex)) for i in range(6) if self._get_font(i).get_char_index(uniindex) != 0]
    if sym == '\\__sqrt__':
        alternatives = alternatives[:-1]
    return alternatives