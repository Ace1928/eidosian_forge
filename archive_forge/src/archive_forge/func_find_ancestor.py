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
def find_ancestor(self, *expression_types: t.Type[E]) -> t.Optional[E]:
    """
        Returns a nearest parent matching expression_types.

        Args:
            expression_types: the expression type(s) to match.

        Returns:
            The parent node.
        """
    ancestor = self.parent
    while ancestor and (not isinstance(ancestor, expression_types)):
        ancestor = ancestor.parent
    return ancestor