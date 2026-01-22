from warnings import warn
from .low_level import MessageType, HeaderFields
from .wrappers import DBusErrorResponse
@on_unhandled.setter
def on_unhandled(self, value):
    warn('Setting on_unhandled is deprecated. Please use the filter() method or simple receive() calls instead.', stacklevel=2)
    self._on_unhandled = value