import inspect
import os
import sys
import unittest
from collections.abc import Sequence
from typing import List
from bpython import inspection
from bpython.test.fodder import encoding_ascii
from bpython.test.fodder import encoding_latin1
from bpython.test.fodder import encoding_utf8
def test_issue_966_static_method(self):

    class Issue966(Sequence):

        @staticmethod
        def cmethod(number: int, lst: List[int]=[]):
            """
                Return a list of numbers

                Example:
                ========
                C.cmethod(1337, [1, 2]) # => [1, 2, 1337]
                """
            return lst + [number]

        @staticmethod
        def bmethod(number, lst):
            """
                Return a list of numbers

                Example:
                ========
                C.cmethod(1337, [1, 2]) # => [1, 2, 1337]
                """
            return lst + [number]
    props = inspection.getfuncprops('bmethod', inspection.getattr_safe(Issue966, 'bmethod'))
    self.assertEqual(props.func, 'bmethod')
    self.assertEqual(props.argspec.args, ['number', 'lst'])
    props = inspection.getfuncprops('cmethod', inspection.getattr_safe(Issue966, 'cmethod'))
    self.assertEqual(props.func, 'cmethod')
    self.assertEqual(props.argspec.args, ['number', 'lst'])
    self.assertEqual(repr(props.argspec.defaults[0]), '[]')