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
def edges2dot(edges, shapes=None, attr=None):
    """
    :param edges: the set (or list) of edges of a directed graph.

    :return dot_string: a representation of 'edges' as a string in the DOT
        graph language, which can be converted to an image by the 'dot' program
        from the Graphviz package, or nltk.parse.dependencygraph.dot2img(dot_string).

    :param shapes: dictionary of strings that trigger a specified shape.
    :param attr: dictionary with global graph attributes

    >>> import nltk
    >>> from nltk.util import edges2dot
    >>> print(edges2dot([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'B')]))
    digraph G {
    "A" -> "B";
    "A" -> "C";
    "B" -> "C";
    "C" -> "B";
    }
    <BLANKLINE>
    """
    if not shapes:
        shapes = dict()
    if not attr:
        attr = dict()
    dot_string = 'digraph G {\n'
    for pair in attr.items():
        dot_string += f'{pair[0]} = {pair[1]};\n'
    for edge in edges:
        for shape in shapes.items():
            for node in range(2):
                if shape[0] in repr(edge[node]):
                    dot_string += f'"{edge[node]}" [shape = {shape[1]}];\n'
        dot_string += f'"{edge[0]}" -> "{edge[1]}";\n'
    dot_string += '}\n'
    return dot_string