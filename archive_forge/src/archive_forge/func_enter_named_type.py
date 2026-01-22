from typing import Any, Collection, List, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import introspection_types, specified_scalar_types
from ...pyutils import did_you_mean, suggestion_list
from . import ASTValidationRule, ValidationContext, SDLValidationContext
def enter_named_type(self, node: NamedTypeNode, _key: Any, parent: Node, _path: Any, ancestors: List[Node]) -> None:
    type_name = node.name.value
    if type_name not in self.existing_types_map and type_name not in self.defined_types:
        try:
            definition_node = ancestors[2]
        except IndexError:
            definition_node = parent
        is_sdl = is_sdl_node(definition_node)
        if is_sdl and type_name in standard_type_names:
            return
        suggested_types = suggestion_list(type_name, list(standard_type_names) + self.type_names if is_sdl else self.type_names)
        self.report_error(GraphQLError(f"Unknown type '{type_name}'." + did_you_mean(suggested_types), node))