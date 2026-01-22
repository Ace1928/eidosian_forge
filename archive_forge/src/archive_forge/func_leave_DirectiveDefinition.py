import json
from .visitor import Visitor, visit
def leave_DirectiveDefinition(self, node, *args):
    return 'directive @{}{} on {}'.format(node.name, wrap('(', join(node.arguments, ', '), ')'), ' | '.join(node.locations))