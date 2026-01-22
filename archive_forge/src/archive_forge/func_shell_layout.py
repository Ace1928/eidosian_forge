import networkx as nx
from networkx.utils import np_random_state
def shell_layout(G, nlist=None, rotate=None, scale=1, center=None, dim=2):
    """Position nodes in concentric circles.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    nlist : list of lists
       List of node lists for each shell.

    rotate : angle in radians (default=pi/len(nlist))
       Angle by which to rotate the starting position of each shell
       relative to the starting position of the previous shell.
       To recreate behavior before v2.5 use rotate=0.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout, currently only dim=2 is supported.
        Other dimension values result in a ValueError.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Raises
    ------
    ValueError
        If dim != 2

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> shells = [[0], [1, 2, 3]]
    >>> pos = nx.shell_layout(G, shells)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    import numpy as np
    if dim != 2:
        raise ValueError('can only handle 2 dimensions')
    G, center = _process_params(G, center, dim)
    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G): center}
    if nlist is None:
        nlist = [list(G)]
    radius_bump = scale / len(nlist)
    if len(nlist[0]) == 1:
        radius = 0.0
    else:
        radius = radius_bump
    if rotate is None:
        rotate = np.pi / len(nlist)
    first_theta = rotate
    npos = {}
    for nodes in nlist:
        theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False, dtype=np.float32) + first_theta
        pos = radius * np.column_stack([np.cos(theta), np.sin(theta)]) + center
        npos.update(zip(nodes, pos))
        radius += radius_bump
        first_theta += rotate
    return npos