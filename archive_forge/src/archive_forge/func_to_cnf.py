from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
def to_cnf(g):
    """Creates a CNF grammar from a general context-free grammar 'g'."""
    g = _unit(_bin(_term(g)))
    return CnfWrapper(g)