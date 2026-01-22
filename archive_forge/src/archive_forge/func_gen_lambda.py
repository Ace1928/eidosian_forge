import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def gen_lambda(self, args, body):
    return ast.Lambda(args=ast.arguments(args=args, vararg=None, kwarg=None, defaults=[]), body=body)