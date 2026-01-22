import sys
import re
import unittest
from curtsies.fmtfuncs import bold, green, magenta, cyan, red, plain
from unittest import mock
from bpython.curtsiesfrontend import interpreter
def test_getsource_works_on_interactively_defined_functions(self):
    source = 'def foo(x):\n    return x + 1\n'
    i = interpreter.Interp()
    i.runsource(source)
    import inspect
    inspected_source = inspect.getsource(i.locals['foo'])
    self.assertEqual(inspected_source, source)