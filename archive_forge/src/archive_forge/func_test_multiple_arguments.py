import unittest2 as unittest
import doctest
import sys
import funcsigs as inspect
def test_multiple_arguments(self):

    def test(a, b=None, *args, **kwargs):
        pass
    self.assertEqual(self.signature(test), ((('a', Ellipsis, Ellipsis, 'positional_or_keyword'), ('b', None, Ellipsis, 'positional_or_keyword'), ('args', Ellipsis, Ellipsis, 'var_positional'), ('kwargs', Ellipsis, Ellipsis, 'var_keyword')), Ellipsis))