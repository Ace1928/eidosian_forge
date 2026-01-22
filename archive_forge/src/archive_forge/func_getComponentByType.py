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
def getComponentByType(self, tagSet, default=noValue, instantiate=True, innerFlag=False):
    """Returns |ASN.1| type component by ASN.1 tag.

        Parameters
        ----------
        tagSet : :py:class:`~pyasn1.type.tag.TagSet`
            Object representing ASN.1 tags to identify one of
            |ASN.1| object component

        Keyword Args
        ------------
        default: :class:`object`
            If set and requested component is a schema object, return the `default`
            object instead of the requested component.

        instantiate: :class:`bool`
            If :obj:`True` (default), inner component will be automatically
            instantiated.
            If :obj:`False` either existing component or the :class:`noValue`
            object will be returned.

        Returns
        -------
        : :py:class:`~pyasn1.type.base.PyAsn1Item`
            a pyasn1 object
        """
    componentValue = self.getComponentByPosition(self.componentType.getPositionByType(tagSet), default=default, instantiate=instantiate)
    if innerFlag and isinstance(componentValue, Set):
        return componentValue.getComponent(innerFlag=True)
    else:
        return componentValue