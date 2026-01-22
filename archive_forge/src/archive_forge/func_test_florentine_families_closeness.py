import pytest
import networkx as nx
def test_florentine_families_closeness(self):
    c = nx.closeness_centrality(self.F)
    d = {'Acciaiuoli': 0.368, 'Albizzi': 0.483, 'Barbadori': 0.4375, 'Bischeri': 0.4, 'Castellani': 0.389, 'Ginori': 0.333, 'Guadagni': 0.467, 'Lamberteschi': 0.326, 'Medici': 0.56, 'Pazzi': 0.286, 'Peruzzi': 0.368, 'Ridolfi': 0.5, 'Salviati': 0.389, 'Strozzi': 0.4375, 'Tornabuoni': 0.483}
    for n in sorted(self.F):
        assert c[n] == pytest.approx(d[n], abs=0.001)