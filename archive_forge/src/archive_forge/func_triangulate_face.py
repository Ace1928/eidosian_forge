from collections import defaultdict
import networkx as nx
def triangulate_face(embedding, v1, v2):
    """Triangulates the face given by half edge (v, w)

    Parameters
    ----------
    embedding : nx.PlanarEmbedding
    v1 : node
        The half-edge (v1, v2) belongs to the face that gets triangulated
    v2 : node
    """
    _, v3 = embedding.next_face_half_edge(v1, v2)
    _, v4 = embedding.next_face_half_edge(v2, v3)
    if v1 in (v2, v3):
        return
    while v1 != v4:
        if embedding.has_edge(v1, v3):
            v1, v2, v3 = (v2, v3, v4)
        else:
            embedding.add_half_edge_cw(v1, v3, v2)
            embedding.add_half_edge_ccw(v3, v1, v2)
            v1, v2, v3 = (v1, v3, v4)
        _, v4 = embedding.next_face_half_edge(v2, v3)