import random
import sys
from . import Nodes
def sum_branchlength(self, root=None, node=None):
    """Add up the branchlengths from root (default self.root) to node.

        sum = sum_branchlength(self,root=None,node=None)
        """
    if root is None:
        root = self.root
    if node is None:
        raise TreeError('Missing node id.')
    blen = 0.0
    while node is not None and node is not root:
        blen += self.node(node).data.branchlength
        node = self.node(node).prev
    return blen