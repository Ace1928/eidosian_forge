import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
@open_file(0, mode='rb')
@nx._dispatch(graphs=None)
def read_gexf(path, node_type=None, relabel=False, version='1.2draft'):
    """Read graph in GEXF format from path.

    "GEXF (Graph Exchange XML Format) is a language for describing
    complex networks structures, their associated data and dynamics" [1]_.

    Parameters
    ----------
    path : file or string
       File or file name to read.
       File names ending in .gz or .bz2 will be decompressed.
    node_type: Python type (default: None)
       Convert node ids to this type if not None.
    relabel : bool (default: False)
       If True relabel the nodes to use the GEXF node "label" attribute
       instead of the node "id" attribute as the NetworkX node label.
    version : string (default: 1.2draft)
    Version of GEFX File Format (see http://gexf.net/schema.html)
       Supported values: "1.1draft", "1.2draft"

    Returns
    -------
    graph: NetworkX graph
        If no parallel edges are found a Graph or DiGraph is returned.
        Otherwise a MultiGraph or MultiDiGraph is returned.

    Notes
    -----
    This implementation does not support mixed graphs (directed and undirected
    edges together).

    References
    ----------
    .. [1] GEXF File Format, http://gexf.net/
    """
    reader = GEXFReader(node_type=node_type, version=version)
    if relabel:
        G = relabel_gexf_graph(reader(path))
    else:
        G = reader(path)
    return G