import unittest2 as unittest
import doctest
import sys
import funcsigs as inspect
def test_single_keyword_argument(self):

    def test(a=None):
        pass
    self.assertEqual(self.signature(test), ((('a', None, Ellipsis, 'positional_or_keyword'),), Ellipsis))