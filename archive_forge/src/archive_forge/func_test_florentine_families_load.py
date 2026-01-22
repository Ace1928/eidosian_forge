import pytest
import networkx as nx
def test_florentine_families_load(self):
    G = self.F
    c = nx.load_centrality(G)
    d = {'Acciaiuoli': 0.0, 'Albizzi': 0.211, 'Barbadori': 0.093, 'Bischeri': 0.104, 'Castellani': 0.055, 'Ginori': 0.0, 'Guadagni': 0.251, 'Lamberteschi': 0.0, 'Medici': 0.522, 'Pazzi': 0.0, 'Peruzzi': 0.022, 'Ridolfi': 0.117, 'Salviati': 0.143, 'Strozzi': 0.106, 'Tornabuoni': 0.09}
    for n in sorted(G):
        assert c[n] == pytest.approx(d[n], abs=0.001)