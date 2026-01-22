import json
from .visitor import Visitor, visit
def leave_VariableDefinition(self, node, *args):
    return node.variable + ': ' + node.type + wrap(' = ', node.default_value)