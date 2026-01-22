import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
def prettyPrintType(self, scope=0):
    scope += 1
    representation = '%s -> %s {\n' % (self.tagSet, self.__class__.__name__)
    for idx, componentType in enumerate(self.componentType.values() or self._componentValues):
        representation += ' ' * scope
        if self.componentType:
            representation += '"%s"' % self.componentType.getNameByPosition(idx)
        else:
            representation += '"%s"' % self._dynamicNames.getNameByPosition(idx)
        representation = '%s = %s\n' % (representation, componentType.prettyPrintType(scope))
    return representation + '\n' + ' ' * (scope - 1) + '}'