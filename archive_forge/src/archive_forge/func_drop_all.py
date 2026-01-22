from contextlib import contextmanager
from itertools import count
from jeepney import HeaderFields, Message, MessageFlag, MessageType
def drop_all(self, exc: Exception=None):
    """Throw an error in any task still waiting for a reply"""
    if exc is None:
        exc = RouterClosed('D-Bus router closed before reply arrived')
    futures, self._futures = (self._futures, {})
    for fut in futures.values():
        fut.set_exception(exc)