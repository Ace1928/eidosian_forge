from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_unhashable(self):

    def foo(a):
        pass
    sig = inspect.signature(foo)
    with self.assertRaisesRegex(TypeError, 'unhashable type'):
        hash(sig)