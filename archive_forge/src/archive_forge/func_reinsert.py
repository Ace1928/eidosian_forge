import warnings
import numpy
def reinsert(self, node, index, bounds):
    """
        Inserts 'node' at the position it had in all lists in [0, 'index'[
        before it was removed. This method assumes that the next and previous
        nodes of the node that is reinserted are in the list.

        """
    for i in range(index):
        node.prev[i].next[i] = node
        node.next[i].prev[i] = node
        if bounds[i] > node.cargo[i]:
            bounds[i] = node.cargo[i]