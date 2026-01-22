from sys import intern
def node_del_attrs(self, node):
    """Remove all attributes that are used for putting this node
        on this type of list.
        """
    try:
        delattr(node, self.next_name)
        delattr(node, self.prev_name)
    except AttributeError:
        pass