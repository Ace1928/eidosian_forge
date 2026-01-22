from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_bind_just_args(self):

    def test(a, b, c):
        return (a, b, c)
    self.assertEqual(self.call(test, 1, 2, 3), (1, 2, 3))
    with self.assertRaisesRegex(TypeError, 'too many positional arguments'):
        self.call(test, 1, 2, 3, 4)
    with self.assertRaisesRegex(TypeError, "'b' parameter lacking default"):
        self.call(test, 1)
    with self.assertRaisesRegex(TypeError, "'a' parameter lacking default"):
        self.call(test)

    def test(a, b, c=10):
        return (a, b, c)
    self.assertEqual(self.call(test, 1, 2, 3), (1, 2, 3))
    self.assertEqual(self.call(test, 1, 2), (1, 2, 10))

    def test(a=1, b=2, c=3):
        return (a, b, c)
    self.assertEqual(self.call(test, a=10, c=13), (10, 2, 13))
    self.assertEqual(self.call(test, a=10), (10, 2, 3))
    self.assertEqual(self.call(test, b=10), (1, 10, 3))