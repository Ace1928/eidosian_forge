from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def walk_in_scope(expression, bfs=True, prune=None):
    """
    Returns a generator object which visits all nodes in the syntrax tree, stopping at
    nodes that start child scopes.

    Args:
        expression (exp.Expression):
        bfs (bool): if set to True the BFS traversal order will be applied,
            otherwise the DFS traversal will be used instead.
        prune ((node, parent, arg_key) -> bool): callable that returns True if
            the generator should stop traversing this branch of the tree.

    Yields:
        tuple[exp.Expression, Optional[exp.Expression], str]: node, parent, arg key
    """
    crossed_scope_boundary = False
    for node in expression.walk(bfs=bfs, prune=lambda n: crossed_scope_boundary or (prune and prune(n))):
        crossed_scope_boundary = False
        yield node
        if node is expression:
            continue
        if isinstance(node, exp.CTE) or (isinstance(node.parent, (exp.From, exp.Join, exp.Subquery)) and (_is_derived_table(node) or isinstance(node, exp.UDTF))) or isinstance(node, exp.UNWRAPPED_QUERIES):
            crossed_scope_boundary = True
            if isinstance(node, (exp.Subquery, exp.UDTF)):
                for key in ('joins', 'laterals', 'pivots'):
                    for arg in node.args.get(key) or []:
                        yield from walk_in_scope(arg, bfs=bfs)