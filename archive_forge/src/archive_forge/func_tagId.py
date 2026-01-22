from pyasn1 import error
@property
def tagId(self):
    """ASN.1 tag ID

        Returns
        -------
        : :py:class:`int`
            Tag ID
        """
    return self.__tagId