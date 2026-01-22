import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
def visitTests(suite, visitor):
    """A foreign method for visiting the tests in a test suite."""
    for test in suite._tests:
        try:
            test.visit(visitor)
        except AttributeError:
            if isinstance(test, unittest.TestCase):
                visitor.visitCase(test)
            elif isinstance(test, unittest.TestSuite):
                visitor.visitSuite(test)
                visitTests(test, visitor)
            else:
                print('unvisitable non-unittest.TestCase element {!r} ({!r})'.format(test, test.__class__))