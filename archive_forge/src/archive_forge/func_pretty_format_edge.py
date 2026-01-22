import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def pretty_format_edge(self, edge, width=None):
    """
        Return a pretty-printed string representation of a given edge
        in this chart.

        :rtype: str
        :param width: The number of characters allotted to each
            index in the sentence.
        """
    if width is None:
        width = 50 // (self.num_leaves() + 1)
    start, end = (edge.start(), edge.end())
    str = '|' + ('.' + ' ' * (width - 1)) * start
    if start == end:
        if edge.is_complete():
            str += '#'
        else:
            str += '>'
    elif edge.is_complete() and edge.span() == (0, self._num_leaves):
        str += '[' + '=' * width * (end - start - 1) + '=' * (width - 1) + ']'
    elif edge.is_complete():
        str += '[' + '-' * width * (end - start - 1) + '-' * (width - 1) + ']'
    else:
        str += '[' + '-' * width * (end - start - 1) + '-' * (width - 1) + '>'
    str += (' ' * (width - 1) + '.') * (self._num_leaves - end)
    return str + '| %s' % edge