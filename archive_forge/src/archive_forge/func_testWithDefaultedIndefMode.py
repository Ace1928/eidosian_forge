import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def testWithDefaultedIndefMode(self):
    self.__initWithDefaulted()
    assert encoder.encode(self.s) == ints2octs((48, 128, 5, 0, 2, 1, 1, 0, 0))