import json
from .visitor import Visitor, visit
def leave_ListValue(self, node, *args):
    return '[' + join(node.values, ', ') + ']'