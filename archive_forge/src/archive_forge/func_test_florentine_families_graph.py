import pytest
import networkx as nx
def test_florentine_families_graph(self):
    """Weighted betweenness centrality:
        Florentine families graph"""
    G = nx.florentine_families_graph()
    b_answer = {'Acciaiuoli': 0.0, 'Albizzi': 0.212, 'Barbadori': 0.093, 'Bischeri': 0.104, 'Castellani': 0.055, 'Ginori': 0.0, 'Guadagni': 0.255, 'Lamberteschi': 0.0, 'Medici': 0.522, 'Pazzi': 0.0, 'Peruzzi': 0.022, 'Ridolfi': 0.114, 'Salviati': 0.143, 'Strozzi': 0.103, 'Tornabuoni': 0.092}
    b = nx.betweenness_centrality(G, weight='weight', normalized=True)
    for n in sorted(G):
        assert b[n] == pytest.approx(b_answer[n], abs=0.001)