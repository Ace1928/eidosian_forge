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
def fetch_rvalue(self) -> List[Token]:
    """Fetch right-hand value of assignment."""
    tokens = []
    while self.fetch_token():
        tokens.append(self.current)
        if self.current == [OP, '(']:
            tokens += self.fetch_until([OP, ')'])
        elif self.current == [OP, '{']:
            tokens += self.fetch_until([OP, '}'])
        elif self.current == [OP, '[']:
            tokens += self.fetch_until([OP, ']'])
        elif self.current == INDENT:
            tokens += self.fetch_until(DEDENT)
        elif self.current == [OP, ';']:
            break
        elif self.current.kind not in (OP, NAME, NUMBER, STRING):
            break
    return tokens