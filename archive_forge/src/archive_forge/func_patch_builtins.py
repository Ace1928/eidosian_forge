import py
import sys
from inspect import CO_VARARGS, CO_VARKEYWORDS, isclass
import traceback
def patch_builtins(assertion=True, compile=True):
    """ put compile and AssertionError builtins to Python's builtins. """
    if assertion:
        from py._code import assertion
        l = oldbuiltins.setdefault('AssertionError', [])
        l.append(py.builtin.builtins.AssertionError)
        py.builtin.builtins.AssertionError = assertion.AssertionError
    if compile:
        l = oldbuiltins.setdefault('compile', [])
        l.append(py.builtin.builtins.compile)
        py.builtin.builtins.compile = py.code.compile