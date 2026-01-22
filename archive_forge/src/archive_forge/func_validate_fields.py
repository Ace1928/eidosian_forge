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
def validate_fields(self, type_: Union[GraphQLObjectType, GraphQLInterfaceType]) -> None:
    fields = type_.fields
    if not fields:
        self.report_error(f'Type {type_.name} must define one or more fields.', [type_.ast_node, *type_.extension_ast_nodes])
    for field_name, field in fields.items():
        self.validate_name(field, field_name)
        if not is_output_type(field.type):
            self.report_error(f'The type of {type_.name}.{field_name} must be Output Type but got: {inspect(field.type)}.', field.ast_node and field.ast_node.type)
        for arg_name, arg in field.args.items():
            self.validate_name(arg, arg_name)
            if not is_input_type(arg.type):
                self.report_error(f'The type of {type_.name}.{field_name}({arg_name}:) must be Input Type but got: {inspect(arg.type)}.', arg.ast_node and arg.ast_node.type)
            if is_required_argument(arg) and arg.deprecation_reason is not None:
                self.report_error(f'Required argument {type_.name}.{field_name}({arg_name}:) cannot be deprecated.', [get_deprecated_directive_node(arg.ast_node), arg.ast_node and arg.ast_node.type])