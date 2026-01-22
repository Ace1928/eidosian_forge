from collections import defaultdict
import networkx as nx
def triangulate_embedding(embedding, fully_triangulate=True):
    """Triangulates the embedding.

    Traverses faces of the embedding and adds edges to a copy of the
    embedding to triangulate it.
    The method also ensures that the resulting graph is 2-connected by adding
    edges if the same vertex is contained twice on a path around a face.

    Parameters
    ----------
    embedding : nx.PlanarEmbedding
        The input graph must contain at least 3 nodes.

    fully_triangulate : bool
        If set to False the face with the most nodes is chooses as outer face.
        This outer face does not get triangulated.

    Returns
    -------
    (embedding, outer_face) : (nx.PlanarEmbedding, list) tuple
        The element `embedding` is a new embedding containing all edges from
        the input embedding and the additional edges to triangulate the graph.
        The element `outer_face` is a list of nodes that lie on the outer face.
        If the graph is fully triangulated these are three arbitrary connected
        nodes.

    """
    if len(embedding.nodes) <= 1:
        return (embedding, list(embedding.nodes))
    embedding = nx.PlanarEmbedding(embedding)
    component_nodes = [next(iter(x)) for x in nx.connected_components(embedding)]
    for i in range(len(component_nodes) - 1):
        v1 = component_nodes[i]
        v2 = component_nodes[i + 1]
        embedding.connect_components(v1, v2)
    outer_face = []
    face_list = []
    edges_visited = set()
    for v in embedding.nodes():
        for w in embedding.neighbors_cw_order(v):
            new_face = make_bi_connected(embedding, v, w, edges_visited)
            if new_face:
                face_list.append(new_face)
                if len(new_face) > len(outer_face):
                    outer_face = new_face
    for face in face_list:
        if face is not outer_face or fully_triangulate:
            triangulate_face(embedding, face[0], face[1])
    if fully_triangulate:
        v1 = outer_face[0]
        v2 = outer_face[1]
        v3 = embedding[v2][v1]['ccw']
        outer_face = [v1, v2, v3]
    return (embedding, outer_face)