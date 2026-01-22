from pyasn1 import error
def tagImplicitly(self, superTag):
    """Return implicitly tagged *TagSet*

        Create a new *TagSet* representing callee *TagSet* implicitly tagged
        with passed tag(s). With implicit tagging mode, new tag(s) replace the
        last existing tag.

        Parameters
        ----------
        superTag: :class:`~pyasn1.type.tag.Tag`
            *Tag* object to tag this *TagSet*

        Returns
        -------
        : :class:`~pyasn1.type.tag.TagSet`
            New *TagSet* object
        """
    if self.__superTags:
        superTag = Tag(superTag.tagClass, self.__superTags[-1].tagFormat, superTag.tagId)
    return self[:-1] + superTag