import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def conll_file_demo():
    print('Mass conll_read demo...')
    graphs = [DependencyGraph(entry) for entry in conll_data2.split('\n\n') if entry]
    for graph in graphs:
        tree = graph.tree()
        print('\n')
        tree.pprint()