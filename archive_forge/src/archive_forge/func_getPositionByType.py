import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
def getPositionByType(self, tagSet):
    """Return field position by its ASN.1 type.

        Parameters
        ----------
        tagSet: :class:`~pysnmp.type.tag.TagSet`
            ASN.1 tag set distinguishing one ASN.1 type from others.

        Returns
        -------
        : :py:class:`int`
            ASN.1 type position in fields set

        Raises
        ------
        : :class:`~pyasn1.error.PyAsn1Error`
            If *tagSet* is not present or ASN.1 types are not unique within callee *NamedTypes*
        """
    try:
        return self.__tagToPosMap[tagSet]
    except KeyError:
        raise error.PyAsn1Error('Type %s not found' % (tagSet,))