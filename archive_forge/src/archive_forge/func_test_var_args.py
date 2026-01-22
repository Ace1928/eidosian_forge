import unittest2 as unittest
import doctest
import sys
import funcsigs as inspect
def test_var_args(self):

    def test(*args):
        pass
    self.assertEqual(self.signature(test), ((('args', Ellipsis, Ellipsis, 'var_positional'),), Ellipsis))