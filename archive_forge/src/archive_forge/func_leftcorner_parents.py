import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def leftcorner_parents(self, cat):
    """
        Return the set of all categories for which the given category
        is a left corner.
        """
    raise NotImplementedError('Not implemented yet')