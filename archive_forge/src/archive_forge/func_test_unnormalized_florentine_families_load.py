import pytest
import networkx as nx
def test_unnormalized_florentine_families_load(self):
    G = self.F
    c = nx.load_centrality(G, normalized=False)
    d = {'Acciaiuoli': 0.0, 'Albizzi': 38.333, 'Barbadori': 17.0, 'Bischeri': 19.0, 'Castellani': 10.0, 'Ginori': 0.0, 'Guadagni': 45.667, 'Lamberteschi': 0.0, 'Medici': 95.0, 'Pazzi': 0.0, 'Peruzzi': 4.0, 'Ridolfi': 21.333, 'Salviati': 26.0, 'Strozzi': 19.333, 'Tornabuoni': 16.333}
    for n in sorted(G):
        assert c[n] == pytest.approx(d[n], abs=0.001)