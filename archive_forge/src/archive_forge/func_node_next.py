from sys import intern
def node_next(self, node):
    try:
        return getattr(node, self.next_name)
    except AttributeError:
        return None