import ast
import collections
import io
import sys
import token
import tokenize
from abc import ABCMeta
from ast import Module, expr, AST
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast, Any, TYPE_CHECKING
from six import iteritems
def last_stmt(node):
    """
  If the given AST node contains multiple statements, return the last one.
  Otherwise, just return the node.
  """
    child_stmts = [child for child in iter_children_func(node)(node) if is_stmt(child) or type(child).__name__ in ('excepthandler', 'ExceptHandler', 'match_case', 'MatchCase', 'TryExcept', 'TryFinally')]
    if child_stmts:
        return last_stmt(child_stmts[-1])
    return node