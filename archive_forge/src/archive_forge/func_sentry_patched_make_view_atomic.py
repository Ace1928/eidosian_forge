from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk import _functools
@_functools.wraps(old_make_view_atomic)
def sentry_patched_make_view_atomic(self, *args, **kwargs):
    callback = old_make_view_atomic(self, *args, **kwargs)
    hub = Hub.current
    integration = hub.get_integration(DjangoIntegration)
    if integration is not None and integration.middleware_spans:
        is_async_view = iscoroutinefunction is not None and wrap_async_view is not None and iscoroutinefunction(callback)
        if is_async_view:
            sentry_wrapped_callback = wrap_async_view(hub, callback)
        else:
            sentry_wrapped_callback = _wrap_sync_view(hub, callback)
    else:
        sentry_wrapped_callback = callback
    return sentry_wrapped_callback