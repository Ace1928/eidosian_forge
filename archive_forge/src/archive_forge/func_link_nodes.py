import weakref
from weakref import ReferenceType
@staticmethod
def link_nodes(previous_node, next_node):
    if next_node:
        next_node.previous_node = previous_node
    if previous_node:
        previous_node.next_node = next_node