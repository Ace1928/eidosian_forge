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
def test_issue_966_freestanding(self):

    def fun(number, lst=[]):
        """
            Return a list of numbers

            Example:
            ========
            C.cmethod(1337, [1, 2]) # => [1, 2, 1337]
            """
        return lst + [number]

    def fun_annotations(number: int, lst: List[int]=[]) -> List[int]:
        """
            Return a list of numbers

            Example:
            ========
            C.cmethod(1337, [1, 2]) # => [1, 2, 1337]
            """
        return lst + [number]
    props = inspection.getfuncprops('fun', fun)
    self.assertEqual(props.func, 'fun')
    self.assertEqual(props.argspec.args, ['number', 'lst'])
    self.assertEqual(repr(props.argspec.defaults[0]), '[]')
    props = inspection.getfuncprops('fun_annotations', fun_annotations)
    self.assertEqual(props.func, 'fun_annotations')
    self.assertEqual(props.argspec.args, ['number', 'lst'])
    self.assertEqual(repr(props.argspec.defaults[0]), '[]')