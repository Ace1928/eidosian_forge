import warnings
from locale import getpreferredencoding
import networkx as nx
from networkx.utils import open_file
Create node positions using :mod:`pydot` and Graphviz.

    Parameters
    ----------
    G : Graph
        NetworkX graph to be laid out.
    prog : string  (default: 'neato')
        Name of the GraphViz command to use for layout.
        Options depend on GraphViz version but may include:
        'dot', 'twopi', 'fdp', 'sfdp', 'circo'
    root : Node from G or None (default: None)
        The node of G from which to start some layout algorithms.

    Returns
    -------
    dict
        Dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.complete_graph(4)
    >>> pos = nx.nx_pydot.pydot_layout(G)
    >>> pos = nx.nx_pydot.pydot_layout(G, prog="dot")

    Notes
    -----
    If you use complex node objects, they may have the same string
    representation and GraphViz could treat them as the same node.
    The layout may assign both nodes a single location. See Issue #1568
    If this occurs in your case, consider relabeling the nodes just
    for the layout computation using something similar to::

        H = nx.convert_node_labels_to_integers(G, label_attribute='node_label')
        H_layout = nx.nx_pydot.pydot_layout(G, prog='dot')
        G_layout = {H.nodes[n]['node_label']: p for n, p in H_layout.items()}

    