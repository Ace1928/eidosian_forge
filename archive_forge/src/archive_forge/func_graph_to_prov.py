import networkx as nx
from prov.model import (
def graph_to_prov(g):
    """
    Convert a `MultiDiGraph
    <https://networkx.github.io/documentation/stable/reference/classes/multidigraph.html>`_
    that was previously produced by :func:`prov_to_graph` back to a
    :class:`~prov.model.ProvDocument`.

    :param g: The graph instance to convert.
    """
    prov_doc = ProvDocument()
    for n in g.nodes():
        if isinstance(n, ProvRecord) and n.bundle is not None:
            prov_doc.add_record(n)
    for _, _, edge_data in g.edges(data=True):
        try:
            relation = edge_data['relation']
            if isinstance(relation, ProvRecord):
                prov_doc.add_record(relation)
        except KeyError:
            pass
    return prov_doc