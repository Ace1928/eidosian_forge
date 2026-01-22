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
@FeatureNew('format strings', '0.58.0')
def evaluate_fstring(self, node: T.Union[mparser.FormatStringNode, mparser.MultilineFormatStringNode]) -> InterpreterObject:

    def replace(match: T.Match[str]) -> str:
        var = str(match.group(1))
        try:
            val = _unholder(self.variables[var])
            if isinstance(val, (list, dict)):
                FeatureNew.single_use('List or dictionary in f-string', '1.3.0', self.subproject, location=self.current_node)
            try:
                return stringifyUserArguments(val, self.subproject)
            except InvalidArguments as e:
                raise InvalidArguments(f'f-string: {str(e)}')
        except KeyError:
            raise InvalidCode(f'Identifier "{var}" does not name a variable.')
    res = re.sub('@([_a-zA-Z][_0-9a-zA-Z]*)@', replace, node.value)
    return self._holderify(res)