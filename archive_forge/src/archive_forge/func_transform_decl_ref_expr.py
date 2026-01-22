from sympy.external import import_module
import os
def transform_decl_ref_expr(self, node):
    """Returns the name of the declaration reference"""
    return node.spelling