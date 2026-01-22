import pytest
import networkx as nx
def test_laplacian_centrality_FF():
    FF = nx.florentine_families_graph()
    d = nx.laplacian_centrality(FF)
    exact = {'Acciaiuoli': 0.0804598, 'Medici': 0.4022989, 'Castellani': 0.1724138, 'Peruzzi': 0.183908, 'Strozzi': 0.2528736, 'Barbadori': 0.137931, 'Ridolfi': 0.2183908, 'Tornabuoni': 0.2183908, 'Albizzi': 0.1954023, 'Salviati': 0.1149425, 'Pazzi': 0.0344828, 'Bischeri': 0.1954023, 'Guadagni': 0.2298851, 'Ginori': 0.045977, 'Lamberteschi': 0.0574713}
    for n, dc in d.items():
        assert exact[n] == pytest.approx(dc, abs=1e-07)