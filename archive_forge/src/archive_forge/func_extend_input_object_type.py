from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def extend_input_object_type(type_: GraphQLInputObjectType) -> GraphQLInputObjectType:
    kwargs = type_.to_kwargs()
    extensions = tuple(type_extensions_map[kwargs['name']])
    return GraphQLInputObjectType(**merge_kwargs(kwargs, fields=lambda: {**{name: GraphQLInputField(**merge_kwargs(field.to_kwargs(), type_=replace_type(field.type))) for name, field in kwargs['fields'].items()}, **build_input_field_map(extensions)}, extension_ast_nodes=kwargs['extension_ast_nodes'] + extensions))