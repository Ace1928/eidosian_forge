import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
def testCmp4(self):
    c = constraint.ConstraintsUnion(constraint.ConstraintsIntersection(constraint.SingleValueConstraint(5)))
    assert self.c1 not in c, '__cmp__() fails'