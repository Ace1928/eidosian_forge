import gzip
import importlib.resources
import os
import os.path
from itertools import islice
import networkx as nx
@nx._dispatch(graphs=None)
def graph_atlas(i):
    """Returns graph number `i` from the Graph Atlas.

    For more information, see :func:`.graph_atlas_g`.

    Parameters
    ----------
    i : int
        The index of the graph from the atlas to get. The graph at index
        0 is assumed to be the null graph.

    Returns
    -------
    list
        A list of :class:`~networkx.Graph` objects, the one at index *i*
        corresponding to the graph *i* in the Graph Atlas.

    See also
    --------
    graph_atlas_g

    Notes
    -----
    The time required by this function increases linearly with the
    argument `i`, since it reads a large file sequentially in order to
    generate the graph [1]_.

    References
    ----------
    .. [1] Ronald C. Read and Robin J. Wilson, *An Atlas of Graphs*.
           Oxford University Press, 1998.

    """
    if not 0 <= i < NUM_GRAPHS:
        raise ValueError(f'index must be between 0 and {NUM_GRAPHS}')
    return next(islice(_generate_graphs(), i, None))