from warnings import warn
from .low_level import MessageType, HeaderFields
from .wrappers import DBusErrorResponse
def subscribe_signal(self, callback, path, interface, member):
    """Add a callback for a signal.
        """
    warn('The subscribe_signal() method is deprecated. Please use the filter() API instead.', stacklevel=2)
    self.signal_callbacks[path, interface, member] = callback