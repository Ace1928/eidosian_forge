from ...error import GraphQLError
from ...language.printer import print_ast
from ...type.definition import is_composite_type
from .base import ValidationRule
@staticmethod
def inline_fragment_on_non_composite_error_message(type):
    return 'Fragment cannot condition on non composite type "{}".'.format(type)