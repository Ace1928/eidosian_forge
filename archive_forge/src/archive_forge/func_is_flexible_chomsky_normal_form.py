import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def is_flexible_chomsky_normal_form(self):
    """
        Return True if all productions are of the forms
        A -> B C, A -> B, or A -> "s".
        """
    return self.is_nonempty() and self.is_nonlexical() and self.is_binarised()