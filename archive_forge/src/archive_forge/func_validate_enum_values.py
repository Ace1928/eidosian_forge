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
def validate_enum_values(self, enum_type: GraphQLEnumType) -> None:
    enum_values = enum_type.values
    if not enum_values:
        self.report_error(f'Enum type {enum_type.name} must define one or more values.', [enum_type.ast_node, *enum_type.extension_ast_nodes])
    for value_name, enum_value in enum_values.items():
        self.validate_name(enum_value, value_name)