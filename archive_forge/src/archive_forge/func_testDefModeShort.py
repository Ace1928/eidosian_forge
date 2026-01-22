import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.der import encoder
from pyasn1.compat.octets import ints2octs
def testDefModeShort(self):
    assert encoder.encode(univ.BitString((1,))) == ints2octs((3, 2, 7, 128))