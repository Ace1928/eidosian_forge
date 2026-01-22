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
def setComponentByName(self, name, value=noValue, verifyConstraints=True, matchTags=True, matchConstraints=True):
    """Assign |ASN.1| type component by name.

        Equivalent to Python :class:`dict` item assignment operation (e.g. `[]`).

        Parameters
        ----------
        name: :class:`str`
            |ASN.1| type component name

        Keyword Args
        ------------
        value: :class:`object` or :py:class:`~pyasn1.type.base.PyAsn1Item` derivative
            A Python value to initialize |ASN.1| component with (if *componentType* is set)
            or ASN.1 value object to assign to |ASN.1| component.
            If `value` is not given, schema object will be set as a component.

        verifyConstraints: :class:`bool`
             If :obj:`False`, skip constraints validation

        matchTags: :class:`bool`
             If :obj:`False`, skip component tags matching

        matchConstraints: :class:`bool`
             If :obj:`False`, skip component constraints matching

        Returns
        -------
        self
        """
    if self._componentTypeLen:
        idx = self.componentType.getPositionByName(name)
    else:
        try:
            idx = self._dynamicNames.getPositionByName(name)
        except KeyError:
            raise error.PyAsn1Error('Name %s not found' % (name,))
    return self.setComponentByPosition(idx, value, verifyConstraints, matchTags, matchConstraints)