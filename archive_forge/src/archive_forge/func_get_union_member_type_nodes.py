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
def get_union_member_type_nodes(union: GraphQLUnionType, type_name: str) -> List[NamedTypeNode]:
    ast_node = union.ast_node
    nodes = union.extension_ast_nodes
    if ast_node is not None:
        nodes = [ast_node, *nodes]
    member_type_nodes: List[NamedTypeNode] = []
    for node in nodes:
        type_nodes = node.types
        if type_nodes:
            member_type_nodes.extend((type_node for type_node in type_nodes if type_node.name.value == type_name))
    return member_type_nodes