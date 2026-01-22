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
def reduce_arguments(self, args: mparser.ArgumentNode, key_resolver: T.Callable[[mparser.BaseNode], str]=default_resolve_key, duplicate_key_error: T.Optional[str]=None) -> T.Tuple[T.List[InterpreterObject], T.Dict[str, InterpreterObject]]:
    assert isinstance(args, mparser.ArgumentNode)
    if args.incorrect_order():
        raise InvalidArguments('All keyword arguments must be after positional arguments.')
    self.argument_depth += 1
    reduced_pos = [self.evaluate_statement(arg) for arg in args.arguments]
    if any((x is None for x in reduced_pos)):
        raise InvalidArguments('At least one value in the arguments is void.')
    reduced_kw: T.Dict[str, InterpreterObject] = {}
    for key, val in args.kwargs.items():
        reduced_key = key_resolver(key)
        assert isinstance(val, mparser.BaseNode)
        reduced_val = self.evaluate_statement(val)
        if reduced_val is None:
            raise InvalidArguments(f'Value of key {reduced_key} is void.')
        self.current_node = key
        if duplicate_key_error and reduced_key in reduced_kw:
            raise InvalidArguments(duplicate_key_error.format(reduced_key))
        reduced_kw[reduced_key] = reduced_val
    self.argument_depth -= 1
    final_kw = self.expand_default_kwargs(reduced_kw)
    return (reduced_pos, final_kw)