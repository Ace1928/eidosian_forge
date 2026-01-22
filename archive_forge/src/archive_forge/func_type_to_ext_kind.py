import re
from functools import partial
from typing import Any, Optional
from ...error import GraphQLError
from ...language import TypeDefinitionNode, TypeExtensionNode
from ...pyutils import did_you_mean, inspect, suggestion_list
from ...type import (
from . import SDLValidationContext, SDLValidationRule
def type_to_ext_kind(type_: Any) -> str:
    if is_scalar_type(type_):
        return 'scalar_type_extension'
    if is_object_type(type_):
        return 'object_type_extension'
    if is_interface_type(type_):
        return 'interface_type_extension'
    if is_union_type(type_):
        return 'union_type_extension'
    if is_enum_type(type_):
        return 'enum_type_extension'
    if is_input_object_type(type_):
        return 'input_object_type_extension'
    raise TypeError(f'Unexpected type: {inspect(type_)}.')