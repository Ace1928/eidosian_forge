import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def num_leaves(self):
    """
        Return the number of words in this chart's sentence.

        :rtype: int
        """
    return self._num_leaves