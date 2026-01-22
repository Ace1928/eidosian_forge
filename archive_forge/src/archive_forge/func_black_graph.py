from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
@sage_method
def black_graph(self):
    """
        Returns the black graph of K.

        If the black graph is disconnected (which can only happen for
        a split link diagram), returns one connected component. The
        edges are labeled by the crossings they correspond to.

        Example::

            sage: K=Link('5_1')
            sage: K.black_graph()
            Subgraph of (): Multi-graph on 2 vertices

        WARNING: While there is also a "white_graph" method, it need
        not be the case that these two graphs are complementary in the
        expected way.
        """
    faces = []
    for x in self.faces():
        l = []
        for y in x:
            l.append((y[0], y[1]))
            l.append((y[0], (y[1] + 1) % 4))
        faces.append(l)
    coords = list()
    for i in range(len(faces) - 1):
        for j in range(i + 1, len(faces)):
            a = set(faces[i])
            b = set(faces[j])
            s = a.union(b)
            for x in range(len(self.crossings)):
                total = set((self.crossings[x][i] for i in range(4)))
                if total.issubset(s):
                    coords.append((tuple(faces[i]), tuple(faces[j]), self.crossings[x]))
    G = graph.Graph(coords, multiedges=True)
    component = G.connected_components(sort=False)[1]
    return G.subgraph(component)