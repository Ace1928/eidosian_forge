import inspect
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.hub import Hub
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import NoOpSpan, Transaction
def set_measurement(name, value, unit=''):
    transaction = Hub.current.scope.transaction
    if transaction is not None:
        transaction.set_measurement(name, value, unit)