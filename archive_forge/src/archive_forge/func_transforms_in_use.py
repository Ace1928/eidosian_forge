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
@property
def transforms_in_use(self):
    """Transformers, excluding logical line transformers if we're in a
        Python line."""
    t = self.physical_line_transforms[:]
    if not self.within_python_line:
        t += [self.assemble_logical_lines] + self.logical_line_transforms
    return t + [self.assemble_python_lines] + self.python_line_transforms