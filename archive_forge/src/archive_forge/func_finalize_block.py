import inspect
import itertools
import re
import tokenize
from collections import OrderedDict
from inspect import Signature
from token import DEDENT, INDENT, NAME, NEWLINE, NUMBER, OP, STRING
from tokenize import COMMENT, NL
from typing import Any, Dict, List, Optional, Tuple
from sphinx.pycode.ast import ast  # for py37 or older
from sphinx.pycode.ast import parse, unparse
def finalize_block(self) -> None:
    """Finalize definition block."""
    definition = self.indents.pop()
    if definition[0] != 'other':
        typ, funcname, start_pos = definition
        end_pos = self.current.end[0] - 1
        while emptyline_re.match(self.get_line(end_pos)):
            end_pos -= 1
        self.add_definition(funcname, (typ, start_pos, end_pos))
        self.context.pop()