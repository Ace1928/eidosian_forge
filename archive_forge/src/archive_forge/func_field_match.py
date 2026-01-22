from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
def field_match(self, node_field, pattern_field):
    """
        Check if two fields match.

        Field match if:
            - If it is a list, all values have to match.
            - If if is a node, recursively check it.
            - Otherwise, check values are equal.
        """
    if isinstance(pattern_field, list):
        return self.check_list(node_field, pattern_field)
    if isinstance(pattern_field, AST):
        return Check(node_field, self.placeholders).visit(pattern_field)
    return Check.strict_eq(pattern_field, node_field)