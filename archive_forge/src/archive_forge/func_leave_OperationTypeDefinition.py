import json
from .visitor import Visitor, visit
def leave_OperationTypeDefinition(self, node, *args):
    return '{}: {}'.format(node.operation, node.type)