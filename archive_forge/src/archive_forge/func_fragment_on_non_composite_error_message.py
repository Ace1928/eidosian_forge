from ...error import GraphQLError
from ...language.printer import print_ast
from ...type.definition import is_composite_type
from .base import ValidationRule
@staticmethod
def fragment_on_non_composite_error_message(frag_name, type):
    return 'Fragment "{}" cannot condition on non composite type "{}".'.format(frag_name, type)