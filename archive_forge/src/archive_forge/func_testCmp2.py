import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
def testCmp2(self):
    assert constraint.SingleValueConstraint(5) not in self.c1, '__cmp__() fails'