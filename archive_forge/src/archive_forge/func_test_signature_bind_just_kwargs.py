from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_bind_just_kwargs(self):

    def test(**kwargs):
        return kwargs
    self.assertEqual(self.call(test), {})
    self.assertEqual(self.call(test, foo='bar', spam='ham'), {'foo': 'bar', 'spam': 'ham'})