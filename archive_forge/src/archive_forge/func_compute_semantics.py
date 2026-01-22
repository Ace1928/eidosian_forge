import itertools
from nltk.ccg.combinator import *
from nltk.ccg.combinator import (
from nltk.ccg.lexicon import Token, fromstring
from nltk.ccg.logic import *
from nltk.parse import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, EdgeI
from nltk.sem.logic import *
from nltk.tree import Tree
def compute_semantics(children, edge):
    if children[0].label()[0].semantics() is None:
        return None
    if len(children) == 2:
        if isinstance(edge.rule(), BackwardCombinator):
            children = [children[1], children[0]]
        combinator = edge.rule()._combinator
        function = children[0].label()[0].semantics()
        argument = children[1].label()[0].semantics()
        if isinstance(combinator, UndirectedFunctionApplication):
            return compute_function_semantics(function, argument)
        elif isinstance(combinator, UndirectedComposition):
            return compute_composition_semantics(function, argument)
        elif isinstance(combinator, UndirectedSubstitution):
            return compute_substitution_semantics(function, argument)
        else:
            raise AssertionError("Unsupported combinator '" + combinator + "'")
    else:
        return compute_type_raised_semantics(children[0].label()[0].semantics())