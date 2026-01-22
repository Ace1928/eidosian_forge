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
def dispatch_custom_completer(self, text):
    """
        .. deprecated:: 8.6
            You can use :meth:`custom_completer_matcher` instead.
        """
    if not self.custom_completers:
        return
    line = self.line_buffer
    if not line.strip():
        return None
    event = SimpleNamespace()
    event.line = line
    event.symbol = text
    cmd = line.split(None, 1)[0]
    event.command = cmd
    event.text_until_cursor = self.text_until_cursor
    if not cmd.startswith(self.magic_escape):
        try_magic = self.custom_completers.s_matches(self.magic_escape + cmd)
    else:
        try_magic = []
    for c in itertools.chain(self.custom_completers.s_matches(cmd), try_magic, self.custom_completers.flat_matches(self.text_until_cursor)):
        try:
            res = c(event)
            if res:
                withcase = [r for r in res if r.startswith(text)]
                if withcase:
                    return withcase
                text_low = text.lower()
                return [r for r in res if r.lower().startswith(text_low)]
        except TryNext:
            pass
        except KeyboardInterrupt:
            '\n                If custom completer take too long,\n                let keyboard interrupt abort and return nothing.\n                '
            break
    return None