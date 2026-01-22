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
@property
def isInconsistent(self):
    """Run necessary checks to ensure |ASN.1| object consistency.

        Default action is to verify |ASN.1| object against constraints imposed
        by `subtypeSpec`.

        Raises
        ------
        :py:class:`~pyasn1.error.PyAsn1tError` on any inconsistencies found
        """
    if self.componentType is noValue or not self.subtypeSpec:
        return False
    if self._componentValues is noValue:
        return True
    mapping = {}
    for idx, value in enumerate(self._componentValues):
        if value is noValue:
            continue
        name = self.componentType.getNameByPosition(idx)
        mapping[name] = value
    try:
        self.subtypeSpec(mapping)
    except error.PyAsn1Error:
        exc = sys.exc_info()[1]
        return exc
    return False