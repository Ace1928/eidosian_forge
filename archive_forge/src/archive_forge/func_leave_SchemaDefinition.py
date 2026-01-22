import json
from .visitor import Visitor, visit
def leave_SchemaDefinition(self, node, *args):
    return join(['schema', join(node.directives, ' '), block(node.operation_types)], ' ')