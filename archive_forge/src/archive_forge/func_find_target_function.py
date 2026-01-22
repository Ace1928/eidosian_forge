from __future__ import annotations
import ast
import inspect
import sys
from collections.abc import Sequence
from functools import partial
from inspect import isclass, isfunction
from types import CodeType, FrameType, FunctionType
from typing import TYPE_CHECKING, Any, Callable, ForwardRef, TypeVar, cast, overload
from warnings import warn
from ._config import CollectionCheckStrategy, ForwardRefPolicy, global_config
from ._exceptions import InstrumentationWarning
from ._functions import TypeCheckFailCallback
from ._transformer import TypeguardTransformer
from ._utils import Unset, function_name, get_stacklevel, is_method_of, unset
def find_target_function(new_code: CodeType, target_path: Sequence[str], firstlineno: int) -> CodeType | None:
    target_name = target_path[0]
    for const in new_code.co_consts:
        if isinstance(const, CodeType):
            if const.co_name == target_name:
                if const.co_firstlineno == firstlineno:
                    return const
                elif len(target_path) > 1:
                    target_code = find_target_function(const, target_path[1:], firstlineno)
                    if target_code:
                        return target_code
    return None