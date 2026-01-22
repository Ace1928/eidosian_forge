import os
import re
import sys
import traceback
import types
import functools
import warnings
from fnmatch import fnmatch, fnmatchcase
from . import case, suite, util
def loadTestsFromTestCase(self, testCaseClass):
    """Return a suite of all test cases contained in testCaseClass"""
    if issubclass(testCaseClass, suite.TestSuite):
        raise TypeError('Test cases should not be derived from TestSuite. Maybe you meant to derive from TestCase?')
    if testCaseClass in (case.TestCase, case.FunctionTestCase):
        testCaseNames = []
    else:
        testCaseNames = self.getTestCaseNames(testCaseClass)
        if not testCaseNames and hasattr(testCaseClass, 'runTest'):
            testCaseNames = ['runTest']
    loaded_suite = self.suiteClass(map(testCaseClass, testCaseNames))
    return loaded_suite