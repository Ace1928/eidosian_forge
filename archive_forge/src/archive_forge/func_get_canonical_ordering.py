from collections import defaultdict
import networkx as nx
def get_canonical_ordering(embedding, outer_face):
    """Returns a canonical ordering of the nodes

    The canonical ordering of nodes (v1, ..., vn) must fulfill the following
    conditions:
    (See Lemma 1 in [2]_)

    - For the subgraph G_k of the input graph induced by v1, ..., vk it holds:
        - 2-connected
        - internally triangulated
        - the edge (v1, v2) is part of the outer face
    - For a node v(k+1) the following holds:
        - The node v(k+1) is part of the outer face of G_k
        - It has at least two neighbors in G_k
        - All neighbors of v(k+1) in G_k lie consecutively on the outer face of
          G_k (excluding the edge (v1, v2)).

    The algorithm used here starts with G_n (containing all nodes). It first
    selects the nodes v1 and v2. And then tries to find the order of the other
    nodes by checking which node can be removed in order to fulfill the
    conditions mentioned above. This is done by calculating the number of
    chords of nodes on the outer face. For more information see [1]_.

    Parameters
    ----------
    embedding : nx.PlanarEmbedding
        The embedding must be triangulated
    outer_face : list
        The nodes on the outer face of the graph

    Returns
    -------
    ordering : list
        A list of tuples `(vk, wp_wq)`. Here `vk` is the node at this position
        in the canonical ordering. The element `wp_wq` is a list of nodes that
        make up the outer face of G_k.

    References
    ----------
    .. [1] Steven Chaplick.
        Canonical Orders of Planar Graphs and (some of) Their Applications 2015
        https://wuecampus2.uni-wuerzburg.de/moodle/pluginfile.php/545727/mod_resource/content/0/vg-ss15-vl03-canonical-orders-druckversion.pdf
    .. [2] M. Chrobak and T.H. Payne:
        A Linear-time Algorithm for Drawing a Planar Graph on a Grid 1989
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.6677

    """
    v1 = outer_face[0]
    v2 = outer_face[1]
    chords = defaultdict(int)
    marked_nodes = set()
    ready_to_pick = set(outer_face)
    outer_face_ccw_nbr = {}
    prev_nbr = v2
    for idx in range(2, len(outer_face)):
        outer_face_ccw_nbr[prev_nbr] = outer_face[idx]
        prev_nbr = outer_face[idx]
    outer_face_ccw_nbr[prev_nbr] = v1
    outer_face_cw_nbr = {}
    prev_nbr = v1
    for idx in range(len(outer_face) - 1, 0, -1):
        outer_face_cw_nbr[prev_nbr] = outer_face[idx]
        prev_nbr = outer_face[idx]

    def is_outer_face_nbr(x, y):
        if x not in outer_face_ccw_nbr:
            return outer_face_cw_nbr[x] == y
        if x not in outer_face_cw_nbr:
            return outer_face_ccw_nbr[x] == y
        return outer_face_ccw_nbr[x] == y or outer_face_cw_nbr[x] == y

    def is_on_outer_face(x):
        return x not in marked_nodes and (x in outer_face_ccw_nbr or x == v1)
    for v in outer_face:
        for nbr in embedding.neighbors_cw_order(v):
            if is_on_outer_face(nbr) and (not is_outer_face_nbr(v, nbr)):
                chords[v] += 1
                ready_to_pick.discard(v)
    canonical_ordering = [None] * len(embedding.nodes())
    canonical_ordering[0] = (v1, [])
    canonical_ordering[1] = (v2, [])
    ready_to_pick.discard(v1)
    ready_to_pick.discard(v2)
    for k in range(len(embedding.nodes()) - 1, 1, -1):
        v = ready_to_pick.pop()
        marked_nodes.add(v)
        wp = None
        wq = None
        nbr_iterator = iter(embedding.neighbors_cw_order(v))
        while True:
            nbr = next(nbr_iterator)
            if nbr in marked_nodes:
                continue
            if is_on_outer_face(nbr):
                if nbr == v1:
                    wp = v1
                elif nbr == v2:
                    wq = v2
                elif outer_face_cw_nbr[nbr] == v:
                    wp = nbr
                else:
                    wq = nbr
            if wp is not None and wq is not None:
                break
        wp_wq = [wp]
        nbr = wp
        while nbr != wq:
            next_nbr = embedding[v][nbr]['ccw']
            wp_wq.append(next_nbr)
            outer_face_cw_nbr[nbr] = next_nbr
            outer_face_ccw_nbr[next_nbr] = nbr
            nbr = next_nbr
        if len(wp_wq) == 2:
            chords[wp] -= 1
            if chords[wp] == 0:
                ready_to_pick.add(wp)
            chords[wq] -= 1
            if chords[wq] == 0:
                ready_to_pick.add(wq)
        else:
            new_face_nodes = set(wp_wq[1:-1])
            for w in new_face_nodes:
                ready_to_pick.add(w)
                for nbr in embedding.neighbors_cw_order(w):
                    if is_on_outer_face(nbr) and (not is_outer_face_nbr(w, nbr)):
                        chords[w] += 1
                        ready_to_pick.discard(w)
                        if nbr not in new_face_nodes:
                            chords[nbr] += 1
                            ready_to_pick.discard(nbr)
        canonical_ordering[k] = (v, wp_wq)
    return canonical_ordering