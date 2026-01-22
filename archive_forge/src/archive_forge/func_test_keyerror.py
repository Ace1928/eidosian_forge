from ..nbs import NetworkBasedStatistic
from ....utils.misc import package_check
import numpy as np
import networkx as nx
import pickle
import pytest
@pytest.mark.skipif(not have_cv, reason='cviewer has to be available')
def test_keyerror(creating_graphs):
    graphlist = creating_graphs
    group1 = graphlist[:3]
    group2 = graphlist[3:]
    nbs = NetworkBasedStatistic()
    nbs.inputs.in_group1 = group1
    nbs.inputs.in_group2 = group2
    nbs.inputs.edge_key = 'Your_edge'
    with pytest.raises(KeyError) as e:
        nbs.run()
    assert 'the graph edges do not have Your_edge attribute' in str(e.value)