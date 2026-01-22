from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def registeredAdapters():
    """Return an iterable of `IAdapterRegistration` instances.

        These registrations describe the current adapter registrations
        in the object.
        """