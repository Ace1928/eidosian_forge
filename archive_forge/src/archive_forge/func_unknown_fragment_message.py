from ...error import GraphQLError
from .base import ValidationRule
@staticmethod
def unknown_fragment_message(fragment_name):
    return 'Unknown fragment "{}".'.format(fragment_name)