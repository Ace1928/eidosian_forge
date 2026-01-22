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
def evaluate_testcase(self, node: mparser.TestCaseClauseNode) -> T.Optional[Disabler]:
    result = self.evaluate_statement(node.condition)
    if isinstance(result, Disabler):
        return result
    if not isinstance(result, ContextManagerObject):
        raise InvalidCode(f'testcase clause {result!r} does not evaluate to a context manager.')
    with result:
        self.evaluate_codeblock(node.block)
    return None