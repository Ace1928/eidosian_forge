from operator import itemgetter
import networkx as nx
def test_most_valuable_edge(self):
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 3), (1, 2, 2), (2, 3, 1)])

    def heaviest(G):
        return max(G.edges(data='weight'), key=itemgetter(2))[:2]
    communities = list(nx.community.girvan_newman(G, heaviest))
    assert len(communities) == 3
    validate_communities(communities[0], [{0}, {1, 2, 3}])
    validate_communities(communities[1], [{0}, {1}, {2, 3}])
    validate_communities(communities[2], [{0}, {1}, {2}, {3}])