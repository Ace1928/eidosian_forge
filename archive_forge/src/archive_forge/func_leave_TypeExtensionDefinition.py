import json
from .visitor import Visitor, visit
def leave_TypeExtensionDefinition(self, node, *args):
    return 'extend ' + node.definition