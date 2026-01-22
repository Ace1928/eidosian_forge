from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_bind_var(self):

    def test(*args, **kwargs):
        return (args, kwargs)
    self.assertEqual(self.call(test), ((), {}))
    self.assertEqual(self.call(test, 1), ((1,), {}))
    self.assertEqual(self.call(test, 1, 2), ((1, 2), {}))
    self.assertEqual(self.call(test, foo='bar'), ((), {'foo': 'bar'}))
    self.assertEqual(self.call(test, 1, foo='bar'), ((1,), {'foo': 'bar'}))
    self.assertEqual(self.call(test, args=10), ((), {'args': 10}))
    self.assertEqual(self.call(test, 1, 2, foo='bar'), ((1, 2), {'foo': 'bar'}))