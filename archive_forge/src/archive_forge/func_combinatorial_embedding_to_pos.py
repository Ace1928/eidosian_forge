from collections import defaultdict
import networkx as nx
def combinatorial_embedding_to_pos(embedding, fully_triangulate=False):
    """Assigns every node a (x, y) position based on the given embedding

    The algorithm iteratively inserts nodes of the input graph in a certain
    order and rearranges previously inserted nodes so that the planar drawing
    stays valid. This is done efficiently by only maintaining relative
    positions during the node placements and calculating the absolute positions
    at the end. For more information see [1]_.

    Parameters
    ----------
    embedding : nx.PlanarEmbedding
        This defines the order of the edges

    fully_triangulate : bool
        If set to True the algorithm adds edges to a copy of the input
        embedding and makes it chordal.

    Returns
    -------
    pos : dict
        Maps each node to a tuple that defines the (x, y) position

    References
    ----------
    .. [1] M. Chrobak and T.H. Payne:
        A Linear-time Algorithm for Drawing a Planar Graph on a Grid 1989
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.6677

    """
    if len(embedding.nodes()) < 4:
        default_positions = [(0, 0), (2, 0), (1, 1)]
        pos = {}
        for i, v in enumerate(embedding.nodes()):
            pos[v] = default_positions[i]
        return pos
    embedding, outer_face = triangulate_embedding(embedding, fully_triangulate)
    left_t_child = {}
    right_t_child = {}
    delta_x = {}
    y_coordinate = {}
    node_list = get_canonical_ordering(embedding, outer_face)
    v1, v2, v3 = (node_list[0][0], node_list[1][0], node_list[2][0])
    delta_x[v1] = 0
    y_coordinate[v1] = 0
    right_t_child[v1] = v3
    left_t_child[v1] = None
    delta_x[v2] = 1
    y_coordinate[v2] = 0
    right_t_child[v2] = None
    left_t_child[v2] = None
    delta_x[v3] = 1
    y_coordinate[v3] = 1
    right_t_child[v3] = v2
    left_t_child[v3] = None
    for k in range(3, len(node_list)):
        vk, contour_neighbors = node_list[k]
        wp = contour_neighbors[0]
        wp1 = contour_neighbors[1]
        wq = contour_neighbors[-1]
        wq1 = contour_neighbors[-2]
        adds_mult_tri = len(contour_neighbors) > 2
        delta_x[wp1] += 1
        delta_x[wq] += 1
        delta_x_wp_wq = sum((delta_x[x] for x in contour_neighbors[1:]))
        delta_x[vk] = (-y_coordinate[wp] + delta_x_wp_wq + y_coordinate[wq]) // 2
        y_coordinate[vk] = (y_coordinate[wp] + delta_x_wp_wq + y_coordinate[wq]) // 2
        delta_x[wq] = delta_x_wp_wq - delta_x[vk]
        if adds_mult_tri:
            delta_x[wp1] -= delta_x[vk]
        right_t_child[wp] = vk
        right_t_child[vk] = wq
        if adds_mult_tri:
            left_t_child[vk] = wp1
            right_t_child[wq1] = None
        else:
            left_t_child[vk] = None
    pos = {}
    pos[v1] = (0, y_coordinate[v1])
    remaining_nodes = [v1]
    while remaining_nodes:
        parent_node = remaining_nodes.pop()
        set_position(parent_node, left_t_child, remaining_nodes, delta_x, y_coordinate, pos)
        set_position(parent_node, right_t_child, remaining_nodes, delta_x, y_coordinate, pos)
    return pos