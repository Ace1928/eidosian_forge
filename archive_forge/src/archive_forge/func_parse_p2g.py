import networkx as nx
from networkx.utils import open_file
@nx._dispatch(graphs=None)
def parse_p2g(lines):
    """Parse p2g format graph from string or iterable.

    Returns
    -------
    MultiDiGraph
    """
    description = next(lines).strip()
    G = nx.MultiDiGraph(name=description, selfloops=True)
    nnodes, nedges = map(int, next(lines).split())
    nodelabel = {}
    nbrs = {}
    for i in range(nnodes):
        n = next(lines).strip()
        nodelabel[i] = n
        G.add_node(n)
        nbrs[n] = map(int, next(lines).split())
    for n in G:
        for nbr in nbrs[n]:
            G.add_edge(n, nodelabel[nbr])
    return G