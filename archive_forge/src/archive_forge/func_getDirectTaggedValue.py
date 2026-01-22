from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def getDirectTaggedValue(tag):
    """
        As for `getTaggedValue`, but never includes inheritance.

        .. versionadded:: 5.0.0
        """