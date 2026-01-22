from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
def test_signature_parameter_kinds(self):
    P = inspect.Parameter
    self.assertTrue(P.POSITIONAL_ONLY < P.POSITIONAL_OR_KEYWORD < P.VAR_POSITIONAL < P.KEYWORD_ONLY < P.VAR_KEYWORD)
    self.assertEqual(str(P.POSITIONAL_ONLY), 'POSITIONAL_ONLY')
    self.assertTrue('POSITIONAL_ONLY' in repr(P.POSITIONAL_ONLY))