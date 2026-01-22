from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import re
import sys
import operator
from types import FunctionType, BuiltinFunctionType
from functools import total_ordering
from io import StringIO
from numba.core import errors, config
from numba.core.utils import (BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS,
from numba.core.errors import (NotDefinedError, RedefinedError,
from numba.core import consts
def dump(self, file=None):
    nofile = file is None
    file = file or StringIO()
    for offset, block in sorted(self.blocks.items()):
        print('label %s:' % (offset,), file=file)
        block.dump(file=file)
    if nofile:
        text = file.getvalue()
        if config.HIGHLIGHT_DUMPS:
            try:
                import pygments
            except ImportError:
                msg = 'Please install pygments to see highlighted dumps'
                raise ValueError(msg)
            else:
                from pygments import highlight
                from numba.misc.dump_style import NumbaIRLexer as lexer
                from numba.misc.dump_style import by_colorscheme
                from pygments.formatters import Terminal256Formatter
                print(highlight(text, lexer(), Terminal256Formatter(style=by_colorscheme())))
        else:
            print(text)