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
def prettyPrint(self, scope=0):
    """Return an object representation string.

        Returns
        -------
        : :class:`str`
            Human-friendly object representation.
        """
    scope += 1
    representation = self.__class__.__name__ + ':\n'
    for idx, componentValue in enumerate(self._componentValues):
        if componentValue is not noValue and componentValue.isValue:
            representation += ' ' * scope
            if self.componentType:
                representation += self.componentType.getNameByPosition(idx)
            else:
                representation += self._dynamicNames.getNameByPosition(idx)
            representation = '%s=%s\n' % (representation, componentValue.prettyPrint(scope))
    return representation