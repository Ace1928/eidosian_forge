import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def tangle_cut(link, cycle):
    """
    Creates two Tangle objects from a given cycle (with no self intersections)
    in the dual graph of a link (inside and outside).  Cycle is given
    as a list of (oriented) edges in the dual graph. Make sure crossings
    are uniquely labeled. Destroys original link.
    """
    sides = {}
    sides[cslabel(cycle[0].interface[0])] = 0
    sides[cslabel(cycle[0].interface[1])] = 1
    side0 = [cycle[0].interface[0]]
    side1 = [cycle[0].interface[1]]
    for i in range(len(cycle) - 1):
        edge1 = cycle[i]
        edge2 = cycle[i + 1]
        edge1_cs1, edge1_cs2 = edge1.interface
        edge2_cs1, edge2_cs2 = edge2.interface
        edge1_cs1_side = sides[cslabel(edge1_cs1)]
        edge1_cs2_side = 1 - edge1_cs1_side
        wrong_face = False
        while True:
            edge1_cs1 = edge1_cs1.next_corner()
            if edge1_cs1 == edge2_cs1:
                sides[cslabel(edge2_cs1)] = edge1_cs2_side
                sides[cslabel(edge2_cs2)] = edge1_cs1_side
                if edge1_cs2_side == 1:
                    side1.append(edge2_cs1)
                    side0.append(edge2_cs2)
                else:
                    side0.append(edge2_cs1)
                    side1.append(edge2_cs2)
                break
            if edge1_cs1 == edge2_cs2:
                sides[cslabel(edge2_cs1)] = edge1_cs1_side
                sides[cslabel(edge2_cs2)] = edge1_cs2_side
                if edge1_cs2_side == 1:
                    side0.append(edge2_cs1)
                    side1.append(edge2_cs2)
                else:
                    side1.append(edge2_cs1)
                    side0.append(edge2_cs2)
                break
            if edge1_cs1.opposite() == edge1_cs2:
                wrong_face = True
                break
        while wrong_face:
            edge1_cs1_side = sides[cslabel(edge1_cs1)]
            edge1_cs2_side = 1 - edge1_cs1_side
            edge1_cs2 = edge1_cs2.next_corner()
            if edge1_cs2 == edge2_cs1:
                sides[cslabel(edge2_cs1)] = edge1_cs1_side
                sides[cslabel(edge2_cs2)] = edge1_cs2_side
                if edge1_cs2_side == 1:
                    side0.append(edge2_cs1)
                    side1.append(edge2_cs2)
                else:
                    side1.append(edge2_cs1)
                    side0.append(edge2_cs2)
                break
            if edge1_cs2 == edge2_cs2:
                sides[cslabel(edge2_cs1)] = edge1_cs2_side
                sides[cslabel(edge2_cs2)] = edge1_cs1_side
                if edge1_cs2_side == 1:
                    side1.append(edge2_cs1)
                    side0.append(edge2_cs2)
                else:
                    side0.append(edge2_cs1)
                    side1.append(edge2_cs2)
                break
            if edge1_cs2.opposite() == edge1_cs1:
                raise Exception('Neither side worked')
    crossing_sides = fill_in_crossings(link, sides)
    n = len(cycle)
    side0[n / 2:] = reversed(side0[n / 2:])
    side1[n / 2:] = reversed(side1[n / 2:])
    crossings0 = [crossing_from_name(link, c) for c in crossing_sides if crossing_sides[c] == 0]
    crossings1 = [crossing_from_name(link, c) for c in crossing_sides if crossing_sides[c] == 1]
    clear_orientations(crossings0)
    clear_orientations(crossings1)
    side0_needs_flip = False
    c, i = side0[0]
    while True:
        next_cep = c.crossing_strands()[(i + 1) % 4]
        c, i = (next_cep.crossing, next_cep.strand_index)
        if (c, i) in side0:
            side0_needs_flip = (c, i) != side0[1]
            break
        c, i = (next_cep.opposite().crossing, next_cep.opposite().strand_index)
    if side0_needs_flip:
        flip(side0)
    else:
        flip(side1)
    return (Tangle(n / 2, crossings0, side0), Tangle(n / 2, crossings1, side1))