import networkx as nx
def test_results_length(self):
    """there is a result for every node"""
    G = small_ego_G()
    disp = nx.dispersion(G)
    disp_Gu = nx.dispersion(G, 'u')
    disp_uv = nx.dispersion(G, 'u', 'h')
    assert len(disp) == len(G)
    assert len(disp_Gu) == len(G) - 1
    assert isinstance(disp_uv, float)