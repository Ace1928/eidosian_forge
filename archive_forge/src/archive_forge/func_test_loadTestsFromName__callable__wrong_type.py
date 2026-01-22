import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromName__callable__wrong_type(self):
    m = types.ModuleType('m')

    def return_wrong():
        return 6
    m.return_wrong = return_wrong
    loader = unittest.TestLoader()
    try:
        suite = loader.loadTestsFromName('return_wrong', m)
    except TypeError:
        pass
    else:
        self.fail('TestLoader.loadTestsFromName failed to raise TypeError')