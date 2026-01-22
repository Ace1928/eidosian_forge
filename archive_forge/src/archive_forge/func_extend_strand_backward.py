from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def extend_strand_backward(kind, strand, start_cep):
    """
    Extend the strand by adding on end_cep and what comes before it
    until you hit a crossing which is not of the given kind
    (over/under).
    """
    cep = start_cep.previous()
    strand.insert(0, start_cep)
    end_cep = strand[-1].next()
    while getattr(cep, 'is_' + kind + '_crossing')():
        if cep.previous() == end_cep:
            break
        strand.insert(0, cep)
        cep = cep.previous()
        if cep == strand[-1]:
            break