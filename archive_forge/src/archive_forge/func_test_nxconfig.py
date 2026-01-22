import collections
import pickle
import pytest
import networkx as nx
from networkx.utils.configs import Config
def test_nxconfig():
    assert isinstance(nx.config.backend_priority, list)
    assert isinstance(nx.config.backends, Config)
    with pytest.raises(TypeError, match='must be a list of backend names'):
        nx.config.backend_priority = 'nx_loopback'
    with pytest.raises(ValueError, match='Unknown backend when setting'):
        nx.config.backend_priority = ['this_almost_certainly_is_not_a_backend']
    with pytest.raises(TypeError, match='must be a Config of backend configs'):
        nx.config.backends = {}
    with pytest.raises(TypeError, match='must be a Config of backend configs'):
        nx.config.backends = Config(plausible_backend_name={})
    with pytest.raises(ValueError, match='Unknown backend when setting'):
        nx.config.backends = Config(this_almost_certainly_is_not_a_backend=Config())
    with pytest.raises(TypeError, match='must be True or False'):
        nx.config.cache_converted_graphs = 'bad value'