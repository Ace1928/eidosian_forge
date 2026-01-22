import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def cycle_finding_demo():
    dg = DependencyGraph(treebank_data)
    print(dg.contains_cycle())
    cyclic_dg = DependencyGraph()
    cyclic_dg.add_node({'word': None, 'deps': [1], 'rel': 'TOP', 'address': 0})
    cyclic_dg.add_node({'word': None, 'deps': [2], 'rel': 'NTOP', 'address': 1})
    cyclic_dg.add_node({'word': None, 'deps': [4], 'rel': 'NTOP', 'address': 2})
    cyclic_dg.add_node({'word': None, 'deps': [1], 'rel': 'NTOP', 'address': 3})
    cyclic_dg.add_node({'word': None, 'deps': [3], 'rel': 'NTOP', 'address': 4})
    print(cyclic_dg.contains_cycle())