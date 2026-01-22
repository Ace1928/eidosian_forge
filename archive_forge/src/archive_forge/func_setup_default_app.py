import weakref
from contextlib import contextmanager
from copy import deepcopy
from kombu.utils.imports import symbol_by_name
from celery import Celery, _state
@contextmanager
def setup_default_app(app, use_trap=False):
    """Setup default app for testing.

    Ensures state is clean after the test returns.
    """
    prev_current_app = _state.get_current_app()
    prev_default_app = _state.default_app
    prev_finalizers = set(_state._on_app_finalizers)
    prev_apps = weakref.WeakSet(_state._apps)
    try:
        if use_trap:
            with set_trap(app):
                yield
        else:
            yield
    finally:
        _state.set_default_app(prev_default_app)
        _state._tls.current_app = prev_current_app
        if app is not prev_current_app:
            app.close()
        _state._on_app_finalizers = prev_finalizers
        _state._apps = prev_apps