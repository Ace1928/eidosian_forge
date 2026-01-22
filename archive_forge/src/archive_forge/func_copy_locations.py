import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def copy_locations(new_node, old_node):
    ast.copy_location(new_node, old_node)
    ast.fix_missing_locations(new_node)