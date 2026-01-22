import json
from .visitor import Visitor, visit
def leave_FloatValue(self, node, *args):
    return node.value