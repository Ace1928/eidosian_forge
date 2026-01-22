import json
from .visitor import Visitor, visit
def leave_EnumTypeDefinition(self, node, *args):
    return 'enum ' + node.name + wrap(' ', join(node.directives, ' ')) + ' ' + block(node.values)