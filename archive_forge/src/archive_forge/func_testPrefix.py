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
def testPrefix(self):
    o = univ.ObjectIdentifier('1.3.6')
    assert o.isPrefixOf((1, 3, 6)), 'isPrefixOf() fails'
    assert o.isPrefixOf((1, 3, 6, 1)), 'isPrefixOf() fails'
    assert not o.isPrefixOf((1, 3)), 'isPrefixOf() fails'