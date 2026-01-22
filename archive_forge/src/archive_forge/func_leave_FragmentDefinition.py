import json
from .visitor import Visitor, visit
def leave_FragmentDefinition(self, node, *args):
    return 'fragment {} on {} '.format(node.name, node.type_condition) + wrap('', join(node.directives, ' '), ' ') + node.selection_set