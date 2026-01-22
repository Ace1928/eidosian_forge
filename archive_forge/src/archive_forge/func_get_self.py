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
def get_self(self) -> Optional[ast.arg]:
    """Returns the name of the first argument if in a function."""
    if self.current_function and self.current_function.args.args:
        return self.current_function.args.args[0]
    elif self.current_function and getattr(self.current_function.args, 'posonlyargs', None):
        return self.current_function.args.posonlyargs[0]
    else:
        return None