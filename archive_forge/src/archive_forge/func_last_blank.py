from __future__ import annotations
from warnings import warn
import ast
import codeop
import io
import re
import sys
import tokenize
import warnings
from typing import List, Tuple, Union, Optional, TYPE_CHECKING
from types import CodeType
from IPython.core.inputtransformer import (leading_indent,
from IPython.utils import tokenutil
from IPython.core.inputtransformer import (ESC_SHELL, ESC_SH_CAP, ESC_HELP,
def last_blank(src):
    """Determine if the input source ends in a blank.

    A blank is either a newline or a line consisting of whitespace.

    Parameters
    ----------
    src : string
        A single or multiline string.
    """
    if not src:
        return False
    ll = src.splitlines()[-1]
    return ll == '' or ll.isspace()