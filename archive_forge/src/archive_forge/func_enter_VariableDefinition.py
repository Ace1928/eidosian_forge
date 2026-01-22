from ...error import GraphQLError
from ...language.printer import print_ast
from ...type.definition import GraphQLNonNull
from ...utils.is_valid_literal_value import is_valid_literal_value
from .base import ValidationRule
def enter_VariableDefinition(self, node, key, parent, path, ancestors):
    name = node.variable.name.value
    default_value = node.default_value
    type = self.context.get_input_type()
    if isinstance(type, GraphQLNonNull) and default_value:
        self.context.report_error(GraphQLError(self.default_for_non_null_arg_message(name, type, type.of_type), [default_value]))
    if type and default_value:
        errors = is_valid_literal_value(type, default_value)
        if errors:
            self.context.report_error(GraphQLError(self.bad_value_for_default_arg_message(name, type, print_ast(default_value), errors), [default_value]))
    return False