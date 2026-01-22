import asyncio
from django.core.handlers.wsgi import WSGIRequest
from sentry_sdk import Hub, _functools
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.utils import capture_internal_exceptions
def patch_channels_asgi_handler_impl(cls):
    import channels
    from sentry_sdk.integrations.django import DjangoIntegration
    if channels.__version__ < '3.0.0':
        old_app = cls.__call__

        async def sentry_patched_asgi_handler(self, receive, send):
            if Hub.current.get_integration(DjangoIntegration) is None:
                return await old_app(self, receive, send)
            middleware = SentryAsgiMiddleware(lambda _scope: old_app.__get__(self, cls), unsafe_context_data=True)
            return await middleware(self.scope)(receive, send)
        cls.__call__ = sentry_patched_asgi_handler
    else:
        patch_django_asgi_handler_impl(cls)