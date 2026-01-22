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
def testGetComponentNoInstantiation(self):
    s = univ.Choice(componentType=namedtype.NamedTypes(namedtype.NamedType('name', univ.OctetString()), namedtype.NamedType('id', univ.Integer())))
    assert s.getComponentByPosition(0, instantiate=False) is univ.noValue
    assert s.getComponentByPosition(1, instantiate=False) is univ.noValue
    assert s.getComponentByName('name', instantiate=False) is univ.noValue
    assert s.getComponentByName('id', instantiate=False) is univ.noValue
    assert s.getComponentByType(univ.OctetString.tagSet, instantiate=False) is univ.noValue
    assert s.getComponentByType(univ.Integer.tagSet, instantiate=False) is univ.noValue
    s[1] = 123
    assert s.getComponentByPosition(1, instantiate=False) is not univ.noValue
    assert s.getComponentByPosition(1, instantiate=False) == 123
    s.clear()
    assert s.getComponentByPosition(1, instantiate=False) is univ.noValue