import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
def testCmp3(self):
    c = constraint.ConstraintsUnion(constraint.ConstraintsIntersection(constraint.SingleValueConstraint(4), constraint.ValueRangeConstraint(2, 4)))
    assert self.c1 in c, '__cmp__() fails'