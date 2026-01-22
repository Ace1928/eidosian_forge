from sys import intern
def node_insert_after(self, node, new_node):
    """Insert the new node after node."""
    assert not self.node_is_on_list(new_node)
    assert node is not new_node
    next = self.node_next(node)
    assert next is not None
    self.node_set_next(node, new_node)
    self.node_set_prev(new_node, node)
    self.node_set_next(new_node, next)
    self.node_set_prev(next, new_node)