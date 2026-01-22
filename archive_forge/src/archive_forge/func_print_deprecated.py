from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_deprecated(reason: Optional[str]) -> str:
    if reason is None:
        return ''
    if reason != DEFAULT_DEPRECATION_REASON:
        ast_value = print_ast(StringValueNode(value=reason))
        return f' @deprecated(reason: {ast_value})'
    return ' @deprecated'