from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def queryDirectTaggedValue(tag, default=None):
    """
        As for `queryTaggedValue`, but never includes inheritance.

        .. versionadded:: 5.0.0
        """