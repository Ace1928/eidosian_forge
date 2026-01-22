from ...error import GraphQLError
from ...language.printer import print_ast
from ...type.definition import GraphQLNonNull
from ...utils.is_valid_literal_value import is_valid_literal_value
from .base import ValidationRule
@staticmethod
def default_for_non_null_arg_message(var_name, type, guess_type):
    return u'Variable "${}" of type "{}" is required and will not use the default value. Perhaps you meant to use type "{}".'.format(var_name, type, guess_type)