from sympy.external import import_module
import os
def transform_return_stmt(self, node):
    """Returns the Return Node for a return statement"""
    return Return(next(node.get_children()).spelling)