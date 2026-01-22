import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import char
from pyasn1.type import univ
from pyasn1.type import constraint
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def testSizeConstraint(self):
    asn1Spec = self.asn1Type(subtypeSpec=constraint.ValueSizeConstraint(1, 1))
    try:
        asn1Spec.clone(self.pythonString)
    except PyAsn1Error:
        pass
    else:
        assert False, 'Size constraint tolerated'
    try:
        asn1Spec.clone(self.pythonString[0])
    except PyAsn1Error:
        assert False, 'Size constraint failed'