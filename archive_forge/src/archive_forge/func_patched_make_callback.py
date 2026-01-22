from __future__ import absolute_import
from sentry_sdk.hub import Hub
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk._types import TYPE_CHECKING
def patched_make_callback(self, *args, **kwargs):
    hub = Hub.current
    integration = hub.get_integration(BottleIntegration)
    prepared_callback = old_make_callback(self, *args, **kwargs)
    if integration is None:
        return prepared_callback
    client = hub.client

    def wrapped_callback(*args, **kwargs):
        try:
            res = prepared_callback(*args, **kwargs)
        except HTTPResponse:
            raise
        except Exception as exception:
            event, hint = event_from_exception(exception, client_options=client.options, mechanism={'type': 'bottle', 'handled': False})
            hub.capture_event(event, hint=hint)
            raise exception
        return res
    return wrapped_callback