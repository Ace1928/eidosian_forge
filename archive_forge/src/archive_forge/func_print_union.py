from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_union(type_: GraphQLUnionType) -> str:
    types = type_.types
    possible_types = ' = ' + ' | '.join((t.name for t in types)) if types else ''
    return print_description(type_) + f'union {type_.name}' + possible_types