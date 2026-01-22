import sys
from sentry_sdk.hub import Hub
from sentry_sdk.utils import event_from_exception
from sentry_sdk._compat import reraise
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
def serverless_function(f=None, flush=True):

    def wrapper(f):

        @wraps(f)
        def inner(*args, **kwargs):
            with Hub(Hub.current) as hub:
                with hub.configure_scope() as scope:
                    scope.clear_breadcrumbs()
                try:
                    return f(*args, **kwargs)
                except Exception:
                    _capture_and_reraise()
                finally:
                    if flush:
                        _flush_client()
        return inner
    if f is None:
        return wrapper
    else:
        return wrapper(f)