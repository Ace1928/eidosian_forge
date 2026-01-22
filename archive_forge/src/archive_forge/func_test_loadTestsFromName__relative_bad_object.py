import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromName__relative_bad_object(self):
    m = types.ModuleType('m')
    m.testcase_1 = object()
    loader = unittest.TestLoader()
    try:
        loader.loadTestsFromName('testcase_1', m)
    except TypeError:
        pass
    else:
        self.fail('Should have raised TypeError')