from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
def visit_AST_or(self, pattern):
    """ Match if any of the or content match with the other node. """
    return any((self.field_match(self.node, value_or) for value_or in pattern.args))