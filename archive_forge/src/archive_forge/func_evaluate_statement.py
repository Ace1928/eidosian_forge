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
def evaluate_statement(self, cur: mparser.BaseNode) -> T.Optional[InterpreterObject]:
    self.current_node = cur
    if isinstance(cur, mparser.FunctionNode):
        return self.function_call(cur)
    elif isinstance(cur, mparser.PlusAssignmentNode):
        self.evaluate_plusassign(cur)
    elif isinstance(cur, mparser.AssignmentNode):
        self.assignment(cur)
    elif isinstance(cur, mparser.MethodNode):
        return self.method_call(cur)
    elif isinstance(cur, mparser.BaseStringNode):
        if isinstance(cur, mparser.MultilineFormatStringNode):
            return self.evaluate_multiline_fstring(cur)
        elif isinstance(cur, mparser.FormatStringNode):
            return self.evaluate_fstring(cur)
        else:
            return self._holderify(cur.value)
    elif isinstance(cur, mparser.BooleanNode):
        return self._holderify(cur.value)
    elif isinstance(cur, mparser.IfClauseNode):
        return self.evaluate_if(cur)
    elif isinstance(cur, mparser.IdNode):
        return self.get_variable(cur.value)
    elif isinstance(cur, mparser.ComparisonNode):
        return self.evaluate_comparison(cur)
    elif isinstance(cur, mparser.ArrayNode):
        return self.evaluate_arraystatement(cur)
    elif isinstance(cur, mparser.DictNode):
        return self.evaluate_dictstatement(cur)
    elif isinstance(cur, mparser.NumberNode):
        return self._holderify(cur.value)
    elif isinstance(cur, mparser.AndNode):
        return self.evaluate_andstatement(cur)
    elif isinstance(cur, mparser.OrNode):
        return self.evaluate_orstatement(cur)
    elif isinstance(cur, mparser.NotNode):
        return self.evaluate_notstatement(cur)
    elif isinstance(cur, mparser.UMinusNode):
        return self.evaluate_uminusstatement(cur)
    elif isinstance(cur, mparser.ArithmeticNode):
        return self.evaluate_arithmeticstatement(cur)
    elif isinstance(cur, mparser.ForeachClauseNode):
        self.evaluate_foreach(cur)
    elif isinstance(cur, mparser.IndexNode):
        return self.evaluate_indexing(cur)
    elif isinstance(cur, mparser.TernaryNode):
        return self.evaluate_ternary(cur)
    elif isinstance(cur, mparser.ContinueNode):
        raise ContinueRequest()
    elif isinstance(cur, mparser.BreakNode):
        raise BreakRequest()
    elif isinstance(cur, mparser.ParenthesizedNode):
        return self.evaluate_statement(cur.inner)
    elif isinstance(cur, mparser.TestCaseClauseNode):
        return self.evaluate_testcase(cur)
    else:
        raise InvalidCode('Unknown statement.')
    return None