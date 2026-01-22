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
def validate_input_fields(self, input_obj: GraphQLInputObjectType) -> None:
    fields = input_obj.fields
    if not fields:
        self.report_error(f'Input Object type {input_obj.name} must define one or more fields.', [input_obj.ast_node, *input_obj.extension_ast_nodes])
    for field_name, field in fields.items():
        self.validate_name(field, field_name)
        if not is_input_type(field.type):
            self.report_error(f'The type of {input_obj.name}.{field_name} must be Input Type but got: {inspect(field.type)}.', field.ast_node.type if field.ast_node else None)
        if is_required_input_field(field) and field.deprecation_reason is not None:
            self.report_error(f'Required input field {input_obj.name}.{field_name} cannot be deprecated.', [get_deprecated_directive_node(field.ast_node), field.ast_node and field.ast_node.type])