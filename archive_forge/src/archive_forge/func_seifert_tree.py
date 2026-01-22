from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def seifert_tree(link):
    """
    The oriented tree corresponding to the complementary regions of
    the Seifert circles.

    >>> K5a2 = [(7,3,8,2),(9,5,0,4),(1,7,2,6),(3,9,4,8),(5,1,6,0)]
    >>> T = seifert_tree(Link(K5a2))
    >>> T == [(frozenset([0]), frozenset([0, 1])), (frozenset([0, 1]), frozenset([1]))]
    True
    """
    circles = seifert_circles(link)
    edges = [[set([n]), set([n])] for n in range(len(circles))]
    for c in link.crossings:
        under, over = c.entry_points()
        under_circle, over_circle = (-1, -1)
        sign = c.sign
        for n, circle in enumerate(circles):
            if under in circle:
                under_circle = n
            if over in circle:
                over_circle = n
            if under_circle > 0 and over_circle > 0:
                break
        if sign == -1:
            connect_head_to_tail(edges[under_circle], edges[over_circle])
        else:
            connect_head_to_tail(edges[over_circle], edges[under_circle])
    for e1, e2 in combinations(edges, 2):
        for i in range(2):
            for j in range(2):
                if len(e1[i].intersection(e2[j])) > 1:
                    connect_vertices(e1, i, e2, j)
    return [(frozenset(e[0]), frozenset(e[1])) for e in edges]