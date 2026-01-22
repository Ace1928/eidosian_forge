import inspect
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import NoOpSpan, Transaction
@hubmethod
def capture_exception(error=None, scope=None, **scope_kwargs):
    return Hub.current.capture_exception(error, scope=scope, **scope_kwargs)