import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def basic_grid_embedding(self, rotate=False):
    """
        Returns the positions of vertices under the grid embedding.
        """
    V = self.chain_coordinates('horizontal')
    H = self.chain_coordinates('vertical')
    return dict(((v, (H[v], V[v])) for v in self.vertices))