import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def is_turn_regular(self):
    if self.exterior:
        return True
    else:
        return kitty_corner(self.turns) is None