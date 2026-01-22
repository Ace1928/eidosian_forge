import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def insert_with_backpointer(self, new_edge, previous_edge, child_edge):
    """
        Add a new edge to the chart, using a pointer to the previous edge.
        """
    cpls = self.child_pointer_lists(previous_edge)
    new_cpls = [cpl + (child_edge,) for cpl in cpls]
    return self.insert(new_edge, *new_cpls)