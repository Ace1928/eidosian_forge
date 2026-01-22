from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def getAdapters(objects, provided):
    """Look for all matching adapters to a provided interface for objects

        Return an iterable of name-adapter pairs for adapters that
        provide the given interface.
        """