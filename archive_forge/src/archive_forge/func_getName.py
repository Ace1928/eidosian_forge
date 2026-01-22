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
def getName(self, innerFlag=False):
    """Return the name of currently assigned component of the |ASN.1| object.

        Returns
        -------
        : :py:class:`str`
            |ASN.1| component name
        """
    if self._currentIdx is None:
        raise error.PyAsn1Error('Component not chosen')
    else:
        if innerFlag:
            c = self._componentValues[self._currentIdx]
            if isinstance(c, Choice):
                return c.getName(innerFlag)
        return self.componentType.getNameByPosition(self._currentIdx)