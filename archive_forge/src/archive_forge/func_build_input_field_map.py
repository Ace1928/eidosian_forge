from collections import defaultdict
from typing import (
from ..language import (
from ..pyutils import inspect, merge_kwargs
from ..type import (
from .value_from_ast import value_from_ast
def build_input_field_map(nodes: Collection[Union[InputObjectTypeDefinitionNode, InputObjectTypeExtensionNode]]) -> GraphQLInputFieldMap:
    input_field_map: GraphQLInputFieldMap = {}
    for node in nodes:
        for field in node.fields or []:
            type_ = cast(GraphQLInputType, get_wrapped_type(field.type))
            input_field_map[field.name.value] = GraphQLInputField(type_=type_, description=field.description.value if field.description else None, default_value=value_from_ast(field.default_value, type_), deprecation_reason=get_deprecation_reason(field), ast_node=field)
    return input_field_map