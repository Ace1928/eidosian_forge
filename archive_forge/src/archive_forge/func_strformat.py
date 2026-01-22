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
def strformat(self, nlines_up=2):
    lines = self.get_lines()
    use_line = self.line
    if self.maybe_decorator:
        tmplines = [''] + lines
        if lines and use_line and ('def ' not in tmplines[use_line]):
            min_line = max(0, use_line)
            max_line = use_line + 10
            selected = tmplines[min_line:max_line]
            index = 0
            for idx, x in enumerate(selected):
                if 'def ' in x:
                    index = idx
                    break
            use_line = use_line + index
    ret = []
    if lines and use_line > 0:

        def count_spaces(string):
            spaces = 0
            for x in itertools.takewhile(str.isspace, str(string)):
                spaces += 1
            return spaces
        selected = lines[max(0, use_line - nlines_up):use_line]
        def_found = False
        for x in selected:
            if 'def ' in x:
                def_found = True
        if not def_found:
            fn_name = None
            for x in reversed(lines[:use_line - 1]):
                if 'def ' in x:
                    fn_name = x
                    break
            if fn_name:
                ret.append(fn_name)
                spaces = count_spaces(x)
                ret.append(' ' * (4 + spaces) + '<source elided>\n')
        if selected:
            ret.extend(selected[:-1])
            ret.append(_termcolor.highlight(selected[-1]))
            spaces = count_spaces(selected[-1])
            ret.append(' ' * spaces + _termcolor.indicate('^'))
    if not ret:
        if not lines:
            ret = '<source missing, REPL/exec in use?>'
        elif use_line <= 0:
            ret = '<source line number missing>'
    err = _termcolor.filename('\nFile "%s", line %d:') + '\n%s'
    tmp = err % (self._get_path(), use_line, _termcolor.code(''.join(ret)))
    return tmp