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
def is_overload(self, decorators: List[ast.expr]) -> bool:
    overload = []
    if self.typing:
        overload.append('%s.overload' % self.typing)
    if self.typing_overload:
        overload.append(self.typing_overload)
    for decorator in decorators:
        try:
            if unparse(decorator) in overload:
                return True
        except NotImplementedError:
            pass
    return False