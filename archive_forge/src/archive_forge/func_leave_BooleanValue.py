import json
from .visitor import Visitor, visit
def leave_BooleanValue(self, node, *args):
    return json.dumps(node.value)