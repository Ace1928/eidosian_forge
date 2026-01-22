import json
from .visitor import Visitor, visit
def leave_InputObjectTypeDefinition(self, node, *args):
    return 'input ' + node.name + wrap(' ', join(node.directives, ' ')) + ' ' + block(node.fields)