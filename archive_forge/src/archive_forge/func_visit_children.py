import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def visit_children(self, node):
    """Visit the contents of a node."""
    return super(UntrustedAstTransformer, self).generic_visit(node)