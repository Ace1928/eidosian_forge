from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def pickup_strand(link, dual_graph, kind, strand):
    """
    Simplify the given (over/under)crossing strand by erasing it from
    the diagram and then finding a path that minimizes the number of
    edges it has to cross over to connect the same endpoints. Returns
    number of crossings removed.
    """
    init_link_cross_count = len(link.crossings)
    G = dual_graph
    startcep = strand[0].previous()
    if startcep == strand[-1]:
        remove_strand(link, strand)
        return len(strand)
    if startcep == strand[-1].next() and startcep.other() in strand:
        remove_strand(link, [startcep] + strand)
        return len(strand)
    crossing_set = set((cep.crossing for cep in strand))
    endpoint = strand[-1].next()
    if endpoint.crossing in crossing_set:
        extend_strand_forward(kind, strand, endpoint)
        return pickup_strand(link, G, kind, strand)
    if startcep.crossing in crossing_set:
        extend_strand_backward(kind, strand, startcep)
        return pickup_strand(link, G, kind, strand)
    edges_crossed = dual_edges(strand, G)
    source = edges_crossed[0][0]
    dest = edges_crossed[-1][0]
    nx.set_edge_attributes(G, 1, 'weight')
    for f0, f1 in edges_crossed:
        G[f0][f1]['weight'] = 0
    path = nx.shortest_path(G, source, dest, weight='weight')
    new_len = sum((G[f0][f1]['weight'] for f0, f1 in zip(path, path[1:])))
    crossingsremoved = len(crossing_set) - new_len
    if crossingsremoved == 0:
        return 0
    newcrossings = []
    removed = remove_strand(link, strand)
    loose_end = startcep.rotate(2)
    for i in range(len(path) - 1):
        label = 'new%d' % i
        f1, f2 = path[i:i + 2]
        edge = G[f1][f2]
        if edge['weight'] > 0:
            cep_to_cross = G[f1][f2]['interface'][f1]
            new_crossing, loose_end = cross(link, cep_to_cross, kind, loose_end, label)
            newcrossings.append(new_crossing)
    ec, ecep = (endpoint.crossing, endpoint.strand_index)
    ec[ecep] = loose_end
    link.crossings.extend(newcrossings)
    active = set()
    for C in removed:
        for i in range(4):
            D = C.adjacent[i][0]
            if D not in removed:
                active.add(D)
    active.update(newcrossings)
    basic_simplify(link, force_build_components=True, to_visit=active)
    final_cross_removed = init_link_cross_count - len(link.crossings)
    assert final_cross_removed >= crossingsremoved
    return final_cross_removed