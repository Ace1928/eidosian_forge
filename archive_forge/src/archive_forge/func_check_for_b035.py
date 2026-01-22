from __future__ import annotations
import ast
import builtins
import itertools
import logging
import math
import re
import sys
import warnings
from collections import namedtuple
from contextlib import suppress
from functools import lru_cache, partial
from keyword import iskeyword
from typing import Dict, List, Set, Union
import attr
import pycodestyle
def check_for_b035(self, node: ast.DictComp):
    """Check that a static key isn't used in a dict comprehension.

        Emit a warning if a likely unchanging key is used - either a constant,
        or a variable that isn't coming from the generator expression.
        """
    if isinstance(node.key, ast.Constant):
        self.errors.append(B035(node.key.lineno, node.key.col_offset, vars=(node.key.value,)))
    elif isinstance(node.key, ast.Name):
        if node.key.id not in self._get_dict_comp_loop_and_named_expr_var_names(node):
            self.errors.append(B035(node.key.lineno, node.key.col_offset, vars=(node.key.id,)))