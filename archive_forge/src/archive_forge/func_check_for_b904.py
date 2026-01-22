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
def check_for_b904(self, node):
    """Checks `raise` without `from` inside an `except` clause.

        In these cases, you should use explicit exception chaining from the
        earlier error, or suppress it with `raise ... from None`.  See
        https://docs.python.org/3/tutorial/errors.html#exception-chaining
        """
    if node.cause is None and node.exc is not None and (not (isinstance(node.exc, ast.Name) and node.exc.id.islower())) and any((isinstance(n, ast.ExceptHandler) for n in self.node_stack)):
        self.errors.append(B904(node.lineno, node.col_offset))