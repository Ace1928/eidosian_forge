from pyasn1 import error
@property
def baseTag(self):
    """Return base ASN.1 tag

        Returns
        -------
        : :class:`~pyasn1.type.tag.Tag`
            Base tag of this *TagSet*
        """
    return self.__baseTag