from __future__ import annotations
import builtins as builtin_mod
import enum
import glob
import inspect
import itertools
import keyword
import os
import re
import string
import sys
import tokenize
import time
import unicodedata
import uuid
import warnings
from ast import literal_eval
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from types import SimpleNamespace
from typing import (
from IPython.core.guarded_eval import guarded_eval, EvaluationContext
from IPython.core.error import TryNext
from IPython.core.inputtransformer2 import ESC_MAGIC
from IPython.core.latex_symbols import latex_symbols, reverse_latex_symbol
from IPython.core.oinspect import InspectColors
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import generics
from IPython.utils.decorators import sphinx_options
from IPython.utils.dir2 import dir2, get_real_method
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.path import ensure_dir_exists
from IPython.utils.process import arg_split
from traitlets import (
from traitlets.config.configurable import Configurable
import __main__
def back_unicode_name_matches(text: str) -> Tuple[str, Sequence[str]]:
    """Match Unicode characters back to Unicode name

    This does  ``☃`` -> ``\\snowman``

    Note that snowman is not a valid python3 combining character but will be expanded.
    Though it will not recombine back to the snowman character by the completion machinery.

    This will not either back-complete standard sequences like \\n, \\b ...

    .. deprecated:: 8.6
        You can use :meth:`back_unicode_name_matcher` instead.

    Returns
    =======

    Return a tuple with two elements:

    - The Unicode character that was matched (preceded with a backslash), or
        empty string,
    - a sequence (of 1), name for the match Unicode character, preceded by
        backslash, or empty if no match.
    """
    if len(text) < 2:
        return ('', ())
    maybe_slash = text[-2]
    if maybe_slash != '\\':
        return ('', ())
    char = text[-1]
    if char in string.ascii_letters or char in ('"', "'"):
        return ('', ())
    try:
        unic = unicodedata.name(char)
        return ('\\' + char, ('\\' + unic,))
    except KeyError:
        pass
    return ('', ())