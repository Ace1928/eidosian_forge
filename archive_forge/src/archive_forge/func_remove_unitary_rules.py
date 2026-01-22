import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
@classmethod
def remove_unitary_rules(cls, grammar):
    """
        Remove nonlexical unitary rules and convert them to
        lexical
        """
    result = []
    unitary = []
    for rule in grammar.productions():
        if len(rule) == 1 and rule.is_nonlexical():
            unitary.append(rule)
        else:
            result.append(rule)
    while unitary:
        rule = unitary.pop(0)
        for item in grammar.productions(lhs=rule.rhs()[0]):
            new_rule = Production(rule.lhs(), item.rhs())
            if len(new_rule) != 1 or new_rule.is_lexical():
                result.append(new_rule)
            else:
                unitary.append(new_rule)
    n_grammar = CFG(grammar.start(), result)
    return n_grammar