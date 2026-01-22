import pytest
import networkx as nx
def test_spiral_layout(self):
    G = self.Gs
    pos_standard = np.array(list(nx.spiral_layout(G, resolution=0.35).values()))
    pos_tighter = np.array(list(nx.spiral_layout(G, resolution=0.34).values()))
    distances = np.linalg.norm(pos_standard[:-1] - pos_standard[1:], axis=1)
    distances_tighter = np.linalg.norm(pos_tighter[:-1] - pos_tighter[1:], axis=1)
    assert sum(distances) > sum(distances_tighter)
    pos_equidistant = np.array(list(nx.spiral_layout(G, equidistant=True).values()))
    distances_equidistant = np.linalg.norm(pos_equidistant[:-1] - pos_equidistant[1:], axis=1)
    assert np.allclose(distances_equidistant[1:], distances_equidistant[-1], atol=0.01)