from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def random_reverse_type_II(link, label1, label2, rebuild=False):
    """
    Randomly crosses two strands, adding two crossings, with labels
    label1 and label2
    """
    faces = link.faces()
    while True:
        face = random.choice(faces)
        if len(face) > 1:
            break
    c, d = random.sample(face, 2)
    reverse_type_II(link, c, d, label1, label2, rebuild=rebuild)