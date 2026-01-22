from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def subscribers(objects, provided):
    """Get subscribers

        Subscribers are returned that provide the provided interface
        and that depend on and are computed from the sequence of
        required objects.
        """