import math
import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import univ
from pyasn1.type import tag
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import error
from pyasn1.compat.octets import str2octs, ints2octs, octs2ints
from pyasn1.error import PyAsn1Error
def testHexStr(self):
    assert univ.OctetString(hexValue='FA9823C43E43510DE3422') == ints2octs((250, 152, 35, 196, 62, 67, 81, 13, 227, 66, 32)), 'hex init fails'