from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.reportviews import DegreeView, EdgeView, NodeView
from networkx.exception import NetworkXError
def bunch_iter(nlist, adj):
    try:
        for n in nlist:
            if n in adj:
                yield n
    except TypeError as err:
        exc, message = (err, err.args[0])
        if 'iter' in message:
            exc = NetworkXError('nbunch is not a node or a sequence of nodes.')
        if 'hashable' in message:
            exc = NetworkXError(f'Node {n} in sequence nbunch is not a valid node.')
        raise exc