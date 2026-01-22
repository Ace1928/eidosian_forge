from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def queryMultiAdapter(objects, interface, name='', default=None):
    """Look for a multi-adapter to an interface for multiple objects

        If a matching adapter cannot be found, returns the default.
        """