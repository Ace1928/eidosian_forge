from __future__ import annotations
from .. import environment, mparser, mesonlib
from .baseobjects import (
from .exceptions import (
from .decorators import FeatureNew
from .disabler import Disabler, is_disabled
from .helpers import default_resolve_key, flatten, resolve_second_level_holders, stringifyUserArguments
from .operator import MesonOperator
from ._unholder import _unholder
import os, copy, re, pathlib
import typing as T
import textwrap
def evaluate_comparison(self, node: mparser.ComparisonNode) -> InterpreterObject:
    val1 = self.evaluate_statement(node.left)
    if val1 is None:
        raise mesonlib.MesonException('Cannot compare a void statement on the left-hand side')
    if isinstance(val1, Disabler):
        return val1
    val2 = self.evaluate_statement(node.right)
    if val2 is None:
        raise mesonlib.MesonException('Cannot compare a void statement on the right-hand side')
    if isinstance(val2, Disabler):
        return val2
    operator = {'in': MesonOperator.IN, 'notin': MesonOperator.NOT_IN, '==': MesonOperator.EQUALS, '!=': MesonOperator.NOT_EQUALS, '>': MesonOperator.GREATER, '<': MesonOperator.LESS, '>=': MesonOperator.GREATER_EQUALS, '<=': MesonOperator.LESS_EQUALS}[node.ctype]
    if operator in (MesonOperator.IN, MesonOperator.NOT_IN):
        val1, val2 = (val2, val1)
    val1.current_node = node
    return self._holderify(val1.operator_call(operator, _unholder(val2)))