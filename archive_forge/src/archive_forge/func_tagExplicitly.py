from pyasn1 import error
def tagExplicitly(self, superTag):
    """Return explicitly tagged *TagSet*

        Create a new *TagSet* representing callee *TagSet* explicitly tagged
        with passed tag(s). With explicit tagging mode, new tags are appended
        to existing tag(s).

        Parameters
        ----------
        superTag: :class:`~pyasn1.type.tag.Tag`
            *Tag* object to tag this *TagSet*

        Returns
        -------
        : :class:`~pyasn1.type.tag.TagSet`
            New *TagSet* object
        """
    if superTag.tagClass == tagClassUniversal:
        raise error.PyAsn1Error("Can't tag with UNIVERSAL class tag")
    if superTag.tagFormat != tagFormatConstructed:
        superTag = Tag(superTag.tagClass, tagFormatConstructed, superTag.tagId)
    return self + superTag