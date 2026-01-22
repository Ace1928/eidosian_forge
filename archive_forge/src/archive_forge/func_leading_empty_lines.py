import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
def leading_empty_lines(lines):
    """Remove leading empty lines

    If the leading lines are empty or contain only whitespace, they will be
    removed.
    """
    if not lines:
        return lines
    for i, line in enumerate(lines):
        if line and (not line.isspace()):
            return lines[i:]
    return lines