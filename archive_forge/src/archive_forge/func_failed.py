from __future__ import absolute_import
import copy
from sentry_sdk import Hub
from sentry_sdk.consts import SPANDATA
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.tracing import Span
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
def failed(self, event):
    hub = Hub.current
    if hub.get_integration(PyMongoIntegration) is None:
        return
    try:
        span = self._ongoing_operations.pop(self._operation_key(event))
        span.set_status('internal_error')
        span.__exit__(None, None, None)
    except KeyError:
        return