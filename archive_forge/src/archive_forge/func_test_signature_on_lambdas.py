from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_on_lambdas(self):
    self.assertEqual(self.signature(lambda a=10: a), ((('a', 10, Ellipsis, 'positional_or_keyword'),), Ellipsis))