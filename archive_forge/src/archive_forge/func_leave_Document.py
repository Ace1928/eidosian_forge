from ...error import GraphQLError
from .base import ValidationRule
def leave_Document(self, node, key, parent, path, ancestors):
    fragment_names_used = set()
    for operation in self.operation_definitions:
        fragments = self.context.get_recursively_referenced_fragments(operation)
        for fragment in fragments:
            fragment_names_used.add(fragment.name.value)
    for fragment_definition in self.fragment_definitions:
        if fragment_definition.name.value not in fragment_names_used:
            self.context.report_error(GraphQLError(self.unused_fragment_message(fragment_definition.name.value), [fragment_definition]))