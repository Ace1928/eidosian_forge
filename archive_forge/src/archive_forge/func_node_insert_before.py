from sys import intern
def node_insert_before(self, node, new_node):
    """Insert the new node before node."""
    assert not self.node_is_on_list(new_node)
    assert node is not new_node
    prev = self.node_prev(node)
    assert prev is not None
    self.node_set_prev(node, new_node)
    self.node_set_next(new_node, node)
    self.node_set_prev(new_node, prev)
    self.node_set_next(prev, new_node)