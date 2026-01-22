from operator import attrgetter
from typing import Any, Collection
from ...error import GraphQLError
from ...language import (
from ...pyutils import group_by
from . import SDLValidationRule
def check_arg_uniqueness(self, parent_name: str, argument_nodes: Collection[InputValueDefinitionNode]) -> VisitorAction:
    seen_args = group_by(argument_nodes, attrgetter('name.value'))
    for arg_name, arg_nodes in seen_args.items():
        if len(arg_nodes) > 1:
            self.report_error(GraphQLError(f"Argument '{parent_name}({arg_name}:)' can only be defined once.", [node.name for node in arg_nodes]))
    return SKIP