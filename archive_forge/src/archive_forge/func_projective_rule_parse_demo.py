from collections import defaultdict
from functools import total_ordering
from itertools import chain
from nltk.grammar import (
from nltk.internals import raise_unorderable_types
from nltk.parse.dependencygraph import DependencyGraph
def projective_rule_parse_demo():
    """
    A demonstration showing the creation and use of a
    ``DependencyGrammar`` to perform a projective dependency
    parse.
    """
    grammar = DependencyGrammar.fromstring("\n    'scratch' -> 'cats' | 'walls'\n    'walls' -> 'the'\n    'cats' -> 'the'\n    ")
    print(grammar)
    pdp = ProjectiveDependencyParser(grammar)
    trees = pdp.parse(['the', 'cats', 'scratch', 'the', 'walls'])
    for tree in trees:
        print(tree)