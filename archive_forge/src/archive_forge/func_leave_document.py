from typing import Any, List
from ...error import GraphQLError
from ...language import (
from . import ASTValidationContext, ASTValidationRule
def leave_document(self, *_args: Any) -> None:
    fragment_names_used = set()
    get_fragments = self.context.get_recursively_referenced_fragments
    for operation in self.operation_defs:
        for fragment in get_fragments(operation):
            fragment_names_used.add(fragment.name.value)
    for fragment_def in self.fragment_defs:
        frag_name = fragment_def.name.value
        if frag_name not in fragment_names_used:
            self.report_error(GraphQLError(f"Fragment '{frag_name}' is never used.", fragment_def))