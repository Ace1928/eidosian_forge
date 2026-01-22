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
def validate_type_implements_ancestors(self, type_: Union[GraphQLObjectType, GraphQLInterfaceType], iface: GraphQLInterfaceType) -> None:
    type_interfaces, iface_interfaces = (type_.interfaces, iface.interfaces)
    for transitive in iface_interfaces:
        if transitive not in type_interfaces:
            self.report_error(f'Type {type_.name} cannot implement {iface.name} because it would create a circular reference.' if transitive is type_ else f'Type {type_.name} must implement {transitive.name} because it is implemented by {iface.name}.', get_all_implements_interface_nodes(iface, transitive) + get_all_implements_interface_nodes(type_, iface))