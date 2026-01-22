from itertools import repeat
from math import sqrt
import networkx as nx
from networkx.classes import set_node_attributes
from networkx.exception import NetworkXError
from networkx.generators.classic import cycle_graph, empty_graph, path_graph
from networkx.relabel import relabel_nodes
from networkx.utils import flatten, nodes_or_number, pairwise
@nx._dispatch(graphs=None)
def hexagonal_lattice_graph(m, n, periodic=False, with_positions=True, create_using=None):
    """Returns an `m` by `n` hexagonal lattice graph.

    The *hexagonal lattice graph* is a graph whose nodes and edges are
    the `hexagonal tiling`_ of the plane.

    The returned graph will have `m` rows and `n` columns of hexagons.
    `Odd numbered columns`_ are shifted up relative to even numbered columns.

    Positions of nodes are computed by default or `with_positions is True`.
    Node positions creating the standard embedding in the plane
    with sidelength 1 and are stored in the node attribute 'pos'.
    `pos = nx.get_node_attributes(G, 'pos')` creates a dict ready for drawing.

    .. _hexagonal tiling: https://en.wikipedia.org/wiki/Hexagonal_tiling
    .. _Odd numbered columns: http://www-cs-students.stanford.edu/~amitp/game-programming/grids/

    Parameters
    ----------
    m : int
        The number of rows of hexagons in the lattice.

    n : int
        The number of columns of hexagons in the lattice.

    periodic : bool
        Whether to make a periodic grid by joining the boundary vertices.
        For this to work `n` must be even and both `n > 1` and `m > 1`.
        The periodic connections create another row and column of hexagons
        so these graphs have fewer nodes as boundary nodes are identified.

    with_positions : bool (default: True)
        Store the coordinates of each node in the graph node attribute 'pos'.
        The coordinates provide a lattice with vertical columns of hexagons
        offset to interleave and cover the plane.
        Periodic positions shift the nodes vertically in a nonlinear way so
        the edges don't overlap so much.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.
        If graph is directed, edges will point up or right.

    Returns
    -------
    NetworkX graph
        The *m* by *n* hexagonal lattice graph.
    """
    G = empty_graph(0, create_using)
    if m == 0 or n == 0:
        return G
    if periodic and (n % 2 == 1 or m < 2 or n < 2):
        msg = 'periodic hexagonal lattice needs m > 1, n > 1 and even n'
        raise NetworkXError(msg)
    M = 2 * m
    rows = range(M + 2)
    cols = range(n + 1)
    col_edges = (((i, j), (i, j + 1)) for i in cols for j in rows[:M + 1])
    row_edges = (((i, j), (i + 1, j)) for i in cols[:n] for j in rows if i % 2 == j % 2)
    G.add_edges_from(col_edges)
    G.add_edges_from(row_edges)
    G.remove_node((0, M + 1))
    G.remove_node((n, (M + 1) * (n % 2)))
    from networkx.algorithms.minors import contracted_nodes
    if periodic:
        for i in cols[:n]:
            G = contracted_nodes(G, (i, 0), (i, M))
        for i in cols[1:]:
            G = contracted_nodes(G, (i, 1), (i, M + 1))
        for j in rows[1:M]:
            G = contracted_nodes(G, (0, j), (n, j))
        G.remove_node((n, M))
    ii = (i for i in cols for j in rows)
    jj = (j for i in cols for j in rows)
    xx = (0.5 + i + i // 2 + j % 2 * (i % 2 - 0.5) for i in cols for j in rows)
    h = sqrt(3) / 2
    if periodic:
        yy = (h * j + 0.01 * i * i for i in cols for j in rows)
    else:
        yy = (h * j for i in cols for j in rows)
    pos = {(i, j): (x, y) for i, j, x, y in zip(ii, jj, xx, yy) if (i, j) in G}
    set_node_attributes(G, pos, 'pos')
    return G