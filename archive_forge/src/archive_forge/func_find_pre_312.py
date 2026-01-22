import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
@classmethod
def find_pre_312(cls, tokens_by_line):
    for line in tokens_by_line:
        assign_ix = _find_assign_op(line)
        if assign_ix is not None and (not line[assign_ix].line.strip().startswith('=')) and (len(line) >= assign_ix + 2) and (line[assign_ix + 1].type == tokenize.ERRORTOKEN):
            ix = assign_ix + 1
            while ix < len(line) and line[ix].type == tokenize.ERRORTOKEN:
                if line[ix].string == '!':
                    return cls(line[ix].start)
                elif not line[ix].string.isspace():
                    break
                ix += 1