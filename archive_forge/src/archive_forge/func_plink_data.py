import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def plink_data(self):
    """
        Returns:
        * a list of vertex positions
        * a list of arrows joining vertices
        * a list of crossings in the format (arrow over, arrow under)
        """
    emb = self.orthogonal_rep().basic_grid_embedding()
    x_max = max((a for a, b in emb.values()))
    y_max = max((b for a, b in emb.values()))
    vertex_positions = []
    for v in self.strand_CEPs:
        if x_max >= y_max:
            a, b = emb[v.crossing]
            b = y_max - b
        else:
            b, a = emb[v.crossing]
        vertex_positions.append((10 * (a + 1), 10 * (b + 1)))
    vert_indices = dict(((v, i) for i, v in enumerate(self.strand_CEPs)))
    arrows, crossings = self.break_into_arrows()
    arrows = [(vert_indices[a[0]], vert_indices[a[-1]]) for a in arrows]
    return (vertex_positions, arrows, crossings)