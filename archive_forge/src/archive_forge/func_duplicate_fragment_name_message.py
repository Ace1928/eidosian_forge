from ...error import GraphQLError
from .base import ValidationRule
@staticmethod
def duplicate_fragment_name_message(field):
    return 'There can only be one fragment named "{}".'.format(field)