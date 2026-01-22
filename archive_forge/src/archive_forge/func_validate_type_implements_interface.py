from operator import attrgetter, itemgetter
from typing import (
from ..error import GraphQLError
from ..pyutils import inspect
from ..language import (
from .definition import (
from ..utilities.type_comparators import is_equal_type, is_type_sub_type_of
from .directives import is_directive, GraphQLDeprecatedDirective
from .introspection import is_introspection_type
from .schema import GraphQLSchema, assert_schema
def validate_type_implements_interface(self, type_: Union[GraphQLObjectType, GraphQLInterfaceType], iface: GraphQLInterfaceType) -> None:
    type_fields, iface_fields = (type_.fields, iface.fields)
    for field_name, iface_field in iface_fields.items():
        type_field = type_fields.get(field_name)
        if not type_field:
            self.report_error(f'Interface field {iface.name}.{field_name} expected but {type_.name} does not provide it.', [iface_field.ast_node, type_.ast_node, *type_.extension_ast_nodes])
            continue
        if not is_type_sub_type_of(self.schema, type_field.type, iface_field.type):
            self.report_error(f'Interface field {iface.name}.{field_name} expects type {iface_field.type} but {type_.name}.{field_name} is type {type_field.type}.', [iface_field.ast_node and iface_field.ast_node.type, type_field.ast_node and type_field.ast_node.type])
        for arg_name, iface_arg in iface_field.args.items():
            type_arg = type_field.args.get(arg_name)
            if not type_arg:
                self.report_error(f'Interface field argument {iface.name}.{field_name}({arg_name}:) expected but {type_.name}.{field_name} does not provide it.', [iface_arg.ast_node, type_field.ast_node])
                continue
            if not is_equal_type(iface_arg.type, type_arg.type):
                self.report_error(f'Interface field argument {iface.name}.{field_name}({arg_name}:) expects type {iface_arg.type} but {type_.name}.{field_name}({arg_name}:) is type {type_arg.type}.', [iface_arg.ast_node and iface_arg.ast_node.type, type_arg.ast_node and type_arg.ast_node.type])
        for arg_name, type_arg in type_field.args.items():
            iface_arg = iface_field.args.get(arg_name)
            if not iface_arg and is_required_argument(type_arg):
                self.report_error(f'Object field {type_.name}.{field_name} includes required argument {arg_name} that is missing from the Interface field {iface.name}.{field_name}.', [type_arg.ast_node, iface_field.ast_node])