import json
from .visitor import Visitor, visit
def leave_ObjectField(self, node, *args):
    return node.name + ': ' + node.value