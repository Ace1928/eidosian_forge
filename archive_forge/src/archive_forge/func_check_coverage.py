import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def check_coverage(self, tokens):
    """
        Check whether the grammar rules cover the given list of tokens.
        If not, then raise an exception.

        :type tokens: list(str)
        """
    missing = [tok for tok in tokens if not self._lexical_index.get(tok)]
    if missing:
        missing = ', '.join((f'{w!r}' for w in missing))
        raise ValueError('Grammar does not cover some of the input words: %r.' % missing)