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
def evaluate_indexing(self, node: mparser.IndexNode) -> InterpreterObject:
    assert isinstance(node, mparser.IndexNode)
    iobject = self.evaluate_statement(node.iobject)
    if iobject is None:
        raise InterpreterException('Tried to evaluate indexing on void.')
    if isinstance(iobject, Disabler):
        return iobject
    index_holder = self.evaluate_statement(node.index)
    if index_holder is None:
        raise InvalidArguments('Cannot use void statement as index.')
    index = _unholder(index_holder)
    iobject.current_node = node
    return self._holderify(iobject.operator_call(MesonOperator.INDEX, index))