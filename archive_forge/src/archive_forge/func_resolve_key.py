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
def resolve_key(key: mparser.BaseNode) -> str:
    if not isinstance(key, mparser.BaseStringNode):
        FeatureNew.single_use('Dictionary entry using non literal key', '0.53.0', self.subproject)
    key_holder = self.evaluate_statement(key)
    if key_holder is None:
        raise InvalidArguments('Key cannot be void.')
    str_key = _unholder(key_holder)
    if not isinstance(str_key, str):
        raise InvalidArguments('Key must be a string')
    return str_key