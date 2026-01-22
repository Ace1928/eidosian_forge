import inspect
import locale
import os
import pydoc
import re
import textwrap
import warnings
from collections import defaultdict, deque
from itertools import chain, combinations, islice, tee
from pprint import pprint
from urllib.request import (
from nltk.collections import *
from nltk.internals import deprecated, raise_unorderable_types, slice_bounds
def unweighted_minimum_spanning_tree(tree, children=iter):
    """
    Output a Minimum Spanning Tree (MST) of an unweighted graph,
    by traversing the nodes of a tree in breadth-first order,
    discarding eventual cycles.

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    >>> import nltk
    >>> from nltk.util import unweighted_minimum_spanning_tree as mst
    >>> wn=nltk.corpus.wordnet
    >>> from pprint import pprint
    >>> pprint(mst(wn.synset('bound.a.01'), lambda s:s.also_sees()))
    [Synset('bound.a.01'),
     [Synset('unfree.a.02'),
      [Synset('confined.a.02')],
      [Synset('dependent.a.01')],
      [Synset('restricted.a.01'), [Synset('classified.a.02')]]]]
    """
    return acyclic_dic2tree(tree, unweighted_minimum_spanning_dict(tree, children))