from __future__ import annotations
import logging
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import while_changing
from sqlglot.optimizer.scope import find_all_in_scope
from sqlglot.optimizer.simplify import flatten, rewrite_between, uniq_sort
def normalization_distance(expression: exp.Expression, dnf: bool=False) -> int:
    """
    The difference in the number of predicates between a given expression and its normalized form.

    This is used as an estimate of the cost of the conversion which is exponential in complexity.

    Example:
        >>> import sqlglot
        >>> expression = sqlglot.parse_one("(a AND b) OR (c AND d)")
        >>> normalization_distance(expression)
        4

    Args:
        expression: The expression to compute the normalization distance for.
        dnf: Whether to check if the expression is in Disjunctive Normal Form (DNF).
            Default: False, i.e. we check if it's in Conjunctive Normal Form (CNF).

    Returns:
        The normalization distance.
    """
    return sum(_predicate_lengths(expression, dnf)) - (sum((1 for _ in expression.find_all(exp.Connector))) + 1)