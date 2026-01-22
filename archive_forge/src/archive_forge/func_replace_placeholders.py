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
def replace_placeholders(expression: Expression, *args, **kwargs) -> Expression:
    """Replace placeholders in an expression.

    Args:
        expression: expression node to be transformed and replaced.
        args: positional names that will substitute unnamed placeholders in the given order.
        kwargs: keyword arguments that will substitute named placeholders.

    Examples:
        >>> from sqlglot import exp, parse_one
        >>> replace_placeholders(
        ...     parse_one("select * from :tbl where ? = ?"),
        ...     exp.to_identifier("str_col"), "b", tbl=exp.to_identifier("foo")
        ... ).sql()
        "SELECT * FROM foo WHERE str_col = 'b'"

    Returns:
        The mapped expression.
    """

    def _replace_placeholders(node: Expression, args, **kwargs) -> Expression:
        if isinstance(node, Placeholder):
            if node.this:
                new_name = kwargs.get(node.this)
                if new_name is not None:
                    return convert(new_name)
            else:
                try:
                    return convert(next(args))
                except StopIteration:
                    pass
        return node
    return expression.transform(_replace_placeholders, iter(args), **kwargs)