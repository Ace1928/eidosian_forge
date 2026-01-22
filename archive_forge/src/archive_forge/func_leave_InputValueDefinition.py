import json
from .visitor import Visitor, visit
def leave_InputValueDefinition(self, node, *args):
    return node.name + ': ' + node.type + wrap(' = ', node.default_value) + wrap(' ', join(node.directives, ' '))