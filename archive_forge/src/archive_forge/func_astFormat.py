import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def astFormat(node):
    return ast.dump(copy.deepcopy(node), annotate_fields=True, include_attributes=True, indent=4)