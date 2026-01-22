import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def protect_unpack_sequence(self, target, value):
    spec = self.gen_unpack_spec(target)
    return ast.Call(func=ast.Name('__rl_unpack_sequence__', ast.Load()), args=[value, spec, ast.Name('__rl_getiter__', ast.Load())], keywords=[])