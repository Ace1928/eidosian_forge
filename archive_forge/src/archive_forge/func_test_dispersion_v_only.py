import networkx as nx
def test_dispersion_v_only(self):
    G = small_ego_G()
    disp_G_h = nx.dispersion(G, v='h', normalized=False)
    disp_G_h_normalized = nx.dispersion(G, v='h', normalized=True)
    assert disp_G_h == {'c': 0, 'f': 0, 'j': 0, 'k': 0, 'u': 4}
    assert disp_G_h_normalized == {'c': 0.0, 'f': 0.0, 'j': 0.0, 'k': 0.0, 'u': 1.0}