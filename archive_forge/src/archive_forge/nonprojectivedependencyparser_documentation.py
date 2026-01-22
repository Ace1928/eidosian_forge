import logging
import math
from nltk.parse.dependencygraph import DependencyGraph

        Parses the input tokens with respect to the parser's grammar.  Parsing
        is accomplished by representing the search-space of possible parses as
        a fully-connected directed graph.  Arcs that would lead to ungrammatical
        parses are removed and a lattice is constructed of length n, where n is
        the number of input tokens, to represent all possible grammatical
        traversals.  All possible paths through the lattice are then enumerated
        to produce the set of non-projective parses.

        param tokens: A list of tokens to parse.
        type tokens: list(str)
        return: An iterator of non-projective parses.
        rtype: iter(DependencyGraph)
        