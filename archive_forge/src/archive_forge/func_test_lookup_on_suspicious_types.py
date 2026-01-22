import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def test_lookup_on_suspicious_types(self):

    class FakeDict:
        pass
    with self.assertRaises(ValueError):
        simple_eval('a[1]', {'a': FakeDict()})

    class TrickyDict(dict):

        def __getitem__(self, index):
            self.fail("doing key lookup isn't safe")
    with self.assertRaises(ValueError):
        simple_eval('a[1]', {'a': TrickyDict()})

    class SchrodingersDict(dict):

        def __getattribute__(inner_self, attr):
            self.fail('doing attribute lookup might have side effects')
    with self.assertRaises(ValueError):
        simple_eval('a[1]', {'a': SchrodingersDict()})

    class SchrodingersCatsDict(dict):

        def __getattr__(inner_self, attr):
            self.fail('doing attribute lookup might have side effects')
    with self.assertRaises(ValueError):
        simple_eval('a[1]', {'a': SchrodingersDict()})