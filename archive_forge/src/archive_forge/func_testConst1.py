import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
def testConst1(self):
    c = constraint.InnerTypeConstraint(constraint.SingleValueConstraint(4))
    try:
        c(4, 32)
    except error.ValueConstraintError:
        assert 0, 'constraint check fails'
    try:
        c(5, 32)
    except error.ValueConstraintError:
        pass
    else:
        assert 0, 'constraint check fails'