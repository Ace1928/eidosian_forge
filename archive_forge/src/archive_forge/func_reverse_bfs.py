from collections import deque
from itertools import islice
import networkx as nx
from ...utils import arbitrary_element
from .utils import (
def reverse_bfs(src):
    """Perform a reverse breadth-first search from src in the residual
        network.
        """
    heights = {src: 0}
    q = deque([(src, 0)])
    while q:
        u, height = q.popleft()
        height += 1
        for v, attr in R_pred[u].items():
            if v not in heights and attr['flow'] < attr['capacity']:
                heights[v] = height
                q.append((v, height))
    return heights