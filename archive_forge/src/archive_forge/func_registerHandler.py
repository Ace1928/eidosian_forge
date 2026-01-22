from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def registerHandler(handler, required=None, name='', info=''):
    """Register a handler.

        A handler is a subscriber that doesn't compute an adapter
        but performs some function when called.

        :param handler:
            The object used to handle some event represented by
            the objects passed to it.

        :param required:
            This is a sequence of specifications for objects to be
            adapted.  If omitted, then the value of the factory's
            ``__component_adapts__`` attribute will be used.  The
            ``__component_adapts__`` attribute is
            normally set using the adapter
            decorator.  If the factory doesn't have a
            ``__component_adapts__`` adapts attribute, then this
            argument is required.

        :param name:
            The handler name.

            Currently, only the empty string is accepted.  Other
            strings will be accepted in the future when support for
            named handlers is added.

        :param info:
           An object that can be converted to a string to provide
           information about the registration.


        A `IRegistered` event is generated with an `IHandlerRegistration`.
        """