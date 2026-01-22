from copy import copy, deepcopy
from typing import (
from ..error import GraphQLError
from ..language import ast, OperationType
from ..pyutils import inspect, is_collection, is_description
from .definition import (
from .directives import GraphQLDirective, specified_directives, is_directive
from .introspection import introspection_types
def remap_named_type(type_: GraphQLNamedType, type_map: TypeMap) -> None:
    """Change all references in the given named type to use this type map."""
    if is_union_type(type_):
        type_ = cast(GraphQLUnionType, type_)
        type_.types = [type_map.get(member_type.name, member_type) for member_type in type_.types]
    elif is_object_type(type_) or is_interface_type(type_):
        type_ = cast(Union[GraphQLObjectType, GraphQLInterfaceType], type_)
        type_.interfaces = [type_map.get(interface_type.name, interface_type) for interface_type in type_.interfaces]
        fields = type_.fields
        for field_name, field in fields.items():
            field = copy(field)
            field.type = remapped_type(field.type, type_map)
            args = field.args
            for arg_name, arg in args.items():
                arg = copy(arg)
                arg.type = remapped_type(arg.type, type_map)
                args[arg_name] = arg
            fields[field_name] = field
    elif is_input_object_type(type_):
        type_ = cast(GraphQLInputObjectType, type_)
        fields = type_.fields
        for field_name, field in fields.items():
            field = copy(field)
            field.type = remapped_type(field.type, type_map)
            fields[field_name] = field