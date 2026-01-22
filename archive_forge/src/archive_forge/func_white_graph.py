from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
@sage_method
def white_graph(self):
    """
        Return the white graph of a non-split link projection.

        This method generates a multigraph whose vertices correspond
        to the faces of the diagram, with an edge joining two
        vertices whenever the corresponding faces contain opposite
        corners at some crossing.  To avoid hashability issues, the
        vertex corresponding to a face is the index of the face in the
        list returned by Link.faces().

        According to the conventions of "Gordon, C. McA. and
        Litherland, R. A, 'On the signature of a link', Inventiones
        math. 47, 23-69 (1978)", in a checkerboard coloring of a link
        diagram the unbounded region is always the first white region.
        Of course, the choice of which region is unbounded is
        arbitrary; it is just a matter of which region on S^2 contains
        the point at infinity.  In this method an equivalent arbitrary
        choice is made by just returning the second component of the
        multigraph, as determined by Graph.connected_components().
        (Empirically, the second component tends to be smaller than
        the first.)

        Note that this may produce a meaningless result in the case of
        a split link diagram.  Consequently if the diagram is split,
        i.e if the multigraph has more than 2 components, a ValueError
        is raised::

            sage: K=Link('5_1')
            sage: K.white_graph()
            Subgraph of (): Multi-graph on 2 vertices

        WARNING: While there is also a "black_graph" method, it need
        not be the case that these two graphs are complementary in the
        expected way.
        """
    face_of = {corner: n for n, face in enumerate(self.faces()) for corner in face}
    edges = []
    for c in self.crossings:
        edges.append((face_of[CrossingStrand(c, 0)], face_of[CrossingStrand(c, 2)], {'crossing': c, 'sign': 1}))
        edges.append((face_of[CrossingStrand(c, 1)], face_of[CrossingStrand(c, 3)], {'crossing': c, 'sign': -1}))
    G = graph.Graph(edges, multiedges=True)
    components = G.connected_components()
    if len(components) > 2:
        raise ValueError('The link diagram is split.')
    return G.subgraph(components[1])