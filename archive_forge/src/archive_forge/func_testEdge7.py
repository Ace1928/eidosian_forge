import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import char
from pyasn1.codec.ber import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def testEdge7(self):
    assert encoder.encode(univ.ObjectIdentifier((2, 100, 3))) == ints2octs((6, 3, 129, 52, 3))