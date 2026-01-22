from pyasn1 import error
def isSuperTagSetOf(self, tagSet):
    """Test type relationship against given *TagSet*

        The callee is considered to be a supertype of given *TagSet*
        tag-wise if all tags in *TagSet* are present in the callee and
        they are in the same order.

        Parameters
        ----------
        tagSet: :class:`~pyasn1.type.tag.TagSet`
            *TagSet* object to evaluate against the callee

        Returns
        -------
        : :py:class:`bool`
            `True` if callee is a supertype of *tagSet*
        """
    if len(tagSet) < self.__lenOfSuperTags:
        return False
    return self.__superTags == tagSet[:self.__lenOfSuperTags]