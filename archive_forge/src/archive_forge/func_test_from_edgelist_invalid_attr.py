import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_from_edgelist_invalid_attr(self):
    pytest.raises(nx.NetworkXError, nx.from_pandas_edgelist, self.df, 0, 'b', 'misspell')
    pytest.raises(nx.NetworkXError, nx.from_pandas_edgelist, self.df, 0, 'b', 1)
    edgeframe = pd.DataFrame([[0, 1], [1, 2], [2, 0]], columns=['s', 't'])
    pytest.raises(nx.NetworkXError, nx.from_pandas_edgelist, edgeframe, 's', 't', True)
    pytest.raises(nx.NetworkXError, nx.from_pandas_edgelist, edgeframe, 's', 't', 'weight')
    pytest.raises(nx.NetworkXError, nx.from_pandas_edgelist, edgeframe, 's', 't', ['weight', 'size'])