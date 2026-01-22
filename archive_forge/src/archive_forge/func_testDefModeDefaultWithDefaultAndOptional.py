import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.der import encoder
from pyasn1.compat.octets import ints2octs
def testDefModeDefaultWithDefaultAndOptional(self):
    s = self.__initDefaultWithDefaultAndOptional()
    assert encoder.encode(s) == ints2octs((48, 11, 48, 9, 4, 4, 116, 101, 115, 116, 2, 1, 123))