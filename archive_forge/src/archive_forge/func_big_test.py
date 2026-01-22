import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def big_test():
    PM = snappy.Manifold()
    while 1:
        M = snappy.HTLinkExteriors.random()
        print('Testing Manifold: ' + M.name())
        _ = test(M, PM)