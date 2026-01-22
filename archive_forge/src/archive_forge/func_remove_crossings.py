from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def remove_crossings(link, eliminate):
    """
    Deletes the given crossings. Assumes that they have already been
    disconnected from the rest of the link, so this just updates
    link.crossings and link.link_components.
    """
    if len(eliminate):
        for C in eliminate:
            link.crossings.remove(C)
        new_components = []
        for component in link.link_components:
            for C in eliminate:
                for cep in C.entry_points():
                    try:
                        component.remove(cep)
                    except ValueError:
                        pass
            if len(component):
                new_components.append(component)
        components_removed = len(link.link_components) - len(new_components)
        link.unlinked_unknot_components += components_removed
        link.link_components = new_components