from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_enum(type_: GraphQLEnumType) -> str:
    values = [print_description(value, '  ', not i) + f'  {name}' + print_deprecated(value.deprecation_reason) for i, (name, value) in enumerate(type_.values.items())]
    return print_description(type_) + f'enum {type_.name}' + print_block(values)