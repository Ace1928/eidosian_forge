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
def unweighted_minimum_spanning_digraph(tree, children=iter, shapes=None, attr=None):
    """

    Build a Minimum Spanning Tree (MST) of an unweighted graph,
    by traversing the nodes of a tree in breadth-first order,
    discarding eventual cycles.

    Return a representation of this MST as a string in the DOT graph language,
    which can be converted to an image by the 'dot' program from the Graphviz
    package, or nltk.parse.dependencygraph.dot2img(dot_string).

    The first argument should be the tree root;
    children should be a function taking as argument a tree node
    and returning an iterator of the node's children.

    >>> import nltk
    >>> wn=nltk.corpus.wordnet
    >>> from nltk.util import unweighted_minimum_spanning_digraph as umsd
    >>> print(umsd(wn.synset('bound.a.01'), lambda s:s.also_sees()))
    digraph G {
    "Synset('bound.a.01')" -> "Synset('unfree.a.02')";
    "Synset('unfree.a.02')" -> "Synset('confined.a.02')";
    "Synset('unfree.a.02')" -> "Synset('dependent.a.01')";
    "Synset('unfree.a.02')" -> "Synset('restricted.a.01')";
    "Synset('restricted.a.01')" -> "Synset('classified.a.02')";
    }
    <BLANKLINE>
    """
    return edges2dot(edge_closure(tree, lambda node: unweighted_minimum_spanning_dict(tree, children)[node]), shapes, attr)