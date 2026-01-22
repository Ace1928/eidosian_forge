from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_fields(type_: Union[GraphQLObjectType, GraphQLInterfaceType]) -> str:
    fields = [print_description(field, '  ', not i) + f'  {name}' + print_args(field.args, '  ') + f': {field.type}' + print_deprecated(field.deprecation_reason) for i, (name, field) in enumerate(type_.fields.items())]
    return print_block(fields)