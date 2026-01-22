from collections import defaultdict
import networkx as nx
def sign_recursive(self, e):
    """Recursive version of :meth:`sign`."""
    if self.ref[e] is not None:
        self.side[e] = self.side[e] * self.sign_recursive(self.ref[e])
        self.ref[e] = None
    return self.side[e]