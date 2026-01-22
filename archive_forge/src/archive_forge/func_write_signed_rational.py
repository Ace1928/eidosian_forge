from __future__ import annotations
import io
import itertools
import logging
import math
import os
import struct
import warnings
from collections.abc import MutableMapping
from fractions import Fraction
from numbers import Number, Rational
from . import ExifTags, Image, ImageFile, ImageOps, ImagePalette, TiffTags
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from .TiffTags import TYPES
@_register_writer(10)
def write_signed_rational(self, *values):
    return b''.join((self._pack('2l', *_limit_signed_rational(frac, 2 ** 31 - 1, -2 ** 31)) for frac in values))