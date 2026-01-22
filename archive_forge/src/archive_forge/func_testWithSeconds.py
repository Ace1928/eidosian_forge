import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def testWithSeconds(self):
    assert encoder.encode(useful.UTCTime('990801120112Z')) == ints2octs((23, 13, 57, 57, 48, 56, 48, 49, 49, 50, 48, 49, 49, 50, 90))