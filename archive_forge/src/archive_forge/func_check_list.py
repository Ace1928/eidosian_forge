from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
def check_list(self, node_list, pattern_list):
    """ Check if list of node are equal. """
    if len(node_list) != len(pattern_list):
        return False
    return all((Check(node_elt, self.placeholders).visit(pattern_elt) for node_elt, pattern_elt in zip(node_list, pattern_list)))