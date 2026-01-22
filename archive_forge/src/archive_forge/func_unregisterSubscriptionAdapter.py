from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def unregisterSubscriptionAdapter(factory=None, required=None, provides=None, name=''):
    """Unregister a subscriber factory.

        :returns:
            A boolean is returned indicating whether the registry was
            changed.  If the given component is None and there is no
            component registered, or if the given component is not
            None and is not registered, then the function returns
            False, otherwise it returns True.

        :param factory:
            This is the object used to compute the adapter. The
            factory can be None, in which case any factories
            registered to implement the given provided interface
            for the given required specifications with the given
            name are unregistered.

        :param required:
            This is a sequence of specifications for objects to be
            adapted.  If omitted, then the value of the factory's
            ``__component_adapts__`` attribute will be used.  The
            ``__component_adapts__`` attribute is
            normally set using the adapter
            decorator.  If the factory doesn't have a
            ``__component_adapts__`` adapts attribute, then this
            argument is required.

        :param provided:
            This is the interface provided by the adapter and
            implemented by the factory.  If the factory is not
            None implements a single interface, then this argument
            is optional and the factory-implemented interface will
            be used.

        :param name:
            The adapter name.

            Currently, only the empty string is accepted.  Other
            strings will be accepted in the future when support for
            named subscribers is added.

        An `IUnregistered` event is generated with an
        `ISubscriptionAdapterRegistration`.
        """