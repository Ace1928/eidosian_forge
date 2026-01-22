import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
@pytest.mark.parametrize(('reportview', 'err_msg_terms'), ((rv.NodeView, 'list(G.nodes'), (rv.NodeDataView, 'list(G.nodes.data'), (rv.EdgeView, 'list(G.edges'), (rv.InEdgeView, 'list(G.in_edges'), (rv.OutEdgeView, 'list(G.edges'), (rv.MultiEdgeView, 'list(G.edges'), (rv.InMultiEdgeView, 'list(G.in_edges'), (rv.OutMultiEdgeView, 'list(G.edges')))
def test_slicing_reportviews(reportview, err_msg_terms):
    G = nx.complete_graph(3)
    view = reportview(G)
    with pytest.raises(nx.NetworkXError) as exc:
        view[0:2]
    errmsg = str(exc.value)
    assert type(view).__name__ in errmsg
    assert err_msg_terms in errmsg