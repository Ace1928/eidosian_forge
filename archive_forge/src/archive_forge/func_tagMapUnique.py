import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
@property
def tagMapUnique(self):
    """Return a *TagMap* object from unique tags and types recursively.

        Return a :class:`~pyasn1.type.tagmap.TagMap` object by
        combining tags from *TagMap* objects of children types and
        associating them with their immediate child type.

        Example
        -------
        .. code-block:: python

           OuterType ::= CHOICE {
               innerType INTEGER
           }

        Calling *.tagMapUnique* on *OuterType* will yield a map like this:

        .. code-block:: python

           Integer.tagSet -> Choice

        Note
        ----

        Duplicate *TagSet* objects found in the tree of children
        types would cause error.
        """
    return self.__uniqueTagMap