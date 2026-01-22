import json
from .visitor import Visitor, visit
def leave_ListType(self, node, *args):
    return '[' + node.type + ']'