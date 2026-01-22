from __future__ import annotations
import datetime
import math
import numbers
import re
import textwrap
import typing as t
from collections import deque
from copy import deepcopy
from enum import auto
from functools import reduce
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.helper import (
from sqlglot.tokens import Token
def iter_expressions(self, reverse: bool=False) -> t.Iterator[Expression]:
    """Yields the key and expression for all arguments, exploding list args."""
    for vs in reversed(tuple(self.args.values())) if reverse else self.args.values():
        if type(vs) is list:
            for v in reversed(vs) if reverse else vs:
                if hasattr(v, 'parent'):
                    yield v
        elif hasattr(vs, 'parent'):
            yield vs