from ...error import GraphQLError
from .base import ValidationRule
def enter_OperationDefinition(self, node, key, parent, path, ancestors):
    return False