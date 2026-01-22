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
@completion_matcher(api_version=1)
def python_matches(self, text: str) -> Iterable[str]:
    """Match attributes or global python names"""
    if '.' in text:
        try:
            matches = self.attr_matches(text)
            if text.endswith('.') and self.omit__names:
                if self.omit__names == 1:
                    no__name = lambda txt: re.match('.*\\.__.*?__', txt) is None
                else:
                    no__name = lambda txt: re.match('\\._.*?', txt[txt.rindex('.'):]) is None
                matches = filter(no__name, matches)
        except NameError:
            matches = []
    else:
        matches = self.global_matches(text)
    return matches