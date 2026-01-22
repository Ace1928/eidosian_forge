import inspect
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import NoOpSpan, Transaction
def get_baggage():
    """
    Returns Baggage either from the active span or from the scope.
    """
    return Hub.current.get_baggage()