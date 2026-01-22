import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def gen_none_node(self):
    return ast.NameConstant(value=None) if hasNameConstant else ast.Name(id='None', ctx=ast.Load())