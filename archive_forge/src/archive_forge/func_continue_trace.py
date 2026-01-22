import inspect
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import NoOpSpan, Transaction
def continue_trace(environ_or_headers, op=None, name=None, source=None):
    """
    Sets the propagation context from environment or headers and returns a transaction.
    """
    return Hub.current.continue_trace(environ_or_headers, op, name, source)