from itertools import repeat
from math import sqrt
import networkx as nx
from networkx.classes import set_node_attributes
from networkx.exception import NetworkXError
from networkx.generators.classic import cycle_graph, empty_graph, path_graph
from networkx.relabel import relabel_nodes
from networkx.utils import flatten, nodes_or_number, pairwise
@nx._dispatch(graphs=None)
def triangular_lattice_graph(m, n, periodic=False, with_positions=True, create_using=None):
    """Returns the $m$ by $n$ triangular lattice graph.

    The `triangular lattice graph`_ is a two-dimensional `grid graph`_ in
    which each square unit has a diagonal edge (each grid unit has a chord).

    The returned graph has $m$ rows and $n$ columns of triangles. Rows and
    columns include both triangles pointing up and down. Rows form a strip
    of constant height. Columns form a series of diamond shapes, staggered
    with the columns on either side. Another way to state the size is that
    the nodes form a grid of `m+1` rows and `(n + 1) // 2` columns.
    The odd row nodes are shifted horizontally relative to the even rows.

    Directed graph types have edges pointed up or right.

    Positions of nodes are computed by default or `with_positions is True`.
    The position of each node (embedded in a euclidean plane) is stored in
    the graph using equilateral triangles with sidelength 1.
    The height between rows of nodes is thus $\\sqrt(3)/2$.
    Nodes lie in the first quadrant with the node $(0, 0)$ at the origin.

    .. _triangular lattice graph: http://mathworld.wolfram.com/TriangularGrid.html
    .. _grid graph: http://www-cs-students.stanford.edu/~amitp/game-programming/grids/
    .. _Triangular Tiling: https://en.wikipedia.org/wiki/Triangular_tiling

    Parameters
    ----------
    m : int
        The number of rows in the lattice.

    n : int
        The number of columns in the lattice.

    periodic : bool (default: False)
        If True, join the boundary vertices of the grid using periodic
        boundary conditions. The join between boundaries is the final row
        and column of triangles. This means there is one row and one column
        fewer nodes for the periodic lattice. Periodic lattices require
        `m >= 3`, `n >= 5` and are allowed but misaligned if `m` or `n` are odd

    with_positions : bool (default: True)
        Store the coordinates of each node in the graph node attribute 'pos'.
        The coordinates provide a lattice with equilateral triangles.
        Periodic positions shift the nodes vertically in a nonlinear way so
        the edges don't overlap so much.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    NetworkX graph
        The *m* by *n* triangular lattice graph.
    """
    H = empty_graph(0, create_using)
    if n == 0 or m == 0:
        return H
    if periodic:
        if n < 5 or m < 3:
            msg = f'm > 2 and n > 4 required for periodic. m={m}, n={n}'
            raise NetworkXError(msg)
    N = (n + 1) // 2
    rows = range(m + 1)
    cols = range(N + 1)
    H.add_edges_from((((i, j), (i + 1, j)) for j in rows for i in cols[:N]))
    H.add_edges_from((((i, j), (i, j + 1)) for j in rows[:m] for i in cols))
    H.add_edges_from((((i, j), (i + 1, j + 1)) for j in rows[1:m:2] for i in cols[:N]))
    H.add_edges_from((((i + 1, j), (i, j + 1)) for j in rows[:m:2] for i in cols[:N]))
    from networkx.algorithms.minors import contracted_nodes
    if periodic is True:
        for i in cols:
            H = contracted_nodes(H, (i, 0), (i, m))
        for j in rows[:m]:
            H = contracted_nodes(H, (0, j), (N, j))
    elif n % 2:
        H.remove_nodes_from(((N, j) for j in rows[1::2]))
    if with_positions:
        ii = (i for i in cols for j in rows)
        jj = (j for i in cols for j in rows)
        xx = (0.5 * (j % 2) + i for i in cols for j in rows)
        h = sqrt(3) / 2
        if periodic:
            yy = (h * j + 0.01 * i * i for i in cols for j in rows)
        else:
            yy = (h * j for i in cols for j in rows)
        pos = {(i, j): (x, y) for i, j, x, y in zip(ii, jj, xx, yy) if (i, j) in H}
        set_node_attributes(H, pos, 'pos')
    return H