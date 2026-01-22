from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def basic_simplify(link, build_components=True, to_visit=None, force_build_components=False):
    """
    Do Reidemeister I and II moves until none are possible.
    """
    if to_visit is None:
        to_visit = set(link.crossings)
    eliminated = set()
    while to_visit:
        crossing = to_visit.pop()
        elim, changed = reidemeister_I_and_II(link, crossing)
        assert not elim.intersection(changed)
        eliminated.update(elim)
        to_visit.difference_update(elim)
        to_visit.update(changed)
    success = len(eliminated) > 0
    if success and build_components or force_build_components:
        component_starts = []
        for component in link.link_components:
            assert len(component) > 0
            if len(component) > 1:
                a, b = component[:2]
            else:
                a = component[0]
                b = a.next()
            if a.strand_label() % 2 == 0:
                component_starts.append(a)
            else:
                component_starts.append(b)
        link._build_components(component_starts)
    return success