import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def fcfg_demo():
    import nltk.data
    g = nltk.data.load('grammars/book_grammars/feat0.fcfg')
    print(g)
    print()