import signal
import weakref
from ... import trace
def install_sighup_handler():
    """Setup a handler for the SIGHUP signal."""
    if getattr(signal, 'SIGHUP', None) is None:
        old_signal = None
    else:
        old_signal = signal.signal(signal.SIGHUP, _sighup_handler)
    old_dict = _setup_on_hangup_dict()
    return (old_signal, old_dict)