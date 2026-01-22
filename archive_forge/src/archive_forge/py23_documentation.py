import decimal as _decimal
import math as _math
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO
from io import StringIO as UnicodeIO
from types import SimpleNamespace
from .textTools import Tag, bytechr, byteord, bytesjoin, strjoin, tobytes, tostr

    Implementation of Python 2 built-in round() function.
    Rounds a number to a given precision in decimal digits (default
    0 digits). The result is a floating point number. Values are rounded
    to the closest multiple of 10 to the power minus ndigits; if two
    multiples are equally close, rounding is done away from 0.
    ndigits may be negative.
    See Python 2 documentation:
    https://docs.python.org/2/library/functions.html?highlight=round#round
    