import weakref
from contextlib import contextmanager
from copy import deepcopy
from kombu.utils.imports import symbol_by_name
from celery import Celery, _state
@contextmanager
def set_trap(app):
    """Contextmanager that installs the trap app.

    The trap means that anything trying to use the current or default app
    will raise an exception.
    """
    trap = Trap()
    prev_tls = _state._tls
    _state.set_default_app(trap)

    class NonTLS:
        current_app = trap
    _state._tls = NonTLS()
    try:
        yield
    finally:
        _state._tls = prev_tls