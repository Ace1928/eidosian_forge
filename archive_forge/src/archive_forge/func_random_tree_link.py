import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def random_tree_link(size):
    """
    Generate two random tree tangles and glues together.  Gives a link of size
    2*size
    """
    return random_tree(size).circular_sum(random_tree(size), 0)