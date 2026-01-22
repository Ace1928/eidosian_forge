import signal
import weakref
from ... import trace
def register_on_hangup(identifier, a_callable):
    """Register for us to call a_callable as part of a graceful shutdown."""
    if _on_sighup is None:
        return
    _on_sighup[identifier] = a_callable