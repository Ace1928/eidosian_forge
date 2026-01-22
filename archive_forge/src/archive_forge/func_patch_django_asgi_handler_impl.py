import asyncio
from django.core.handlers.wsgi import WSGIRequest
from sentry_sdk import Hub, _functools
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.utils import capture_internal_exceptions
def patch_django_asgi_handler_impl(cls):
    from sentry_sdk.integrations.django import DjangoIntegration
    old_app = cls.__call__

    async def sentry_patched_asgi_handler(self, scope, receive, send):
        hub = Hub.current
        integration = hub.get_integration(DjangoIntegration)
        if integration is None:
            return await old_app(self, scope, receive, send)
        middleware = SentryAsgiMiddleware(old_app.__get__(self, cls), unsafe_context_data=True)._run_asgi3
        return await middleware(scope, receive, send)
    cls.__call__ = sentry_patched_asgi_handler
    modern_django_asgi_support = hasattr(cls, 'create_request')
    if modern_django_asgi_support:
        old_create_request = cls.create_request

        def sentry_patched_create_request(self, *args, **kwargs):
            hub = Hub.current
            integration = hub.get_integration(DjangoIntegration)
            if integration is None:
                return old_create_request(self, *args, **kwargs)
            with hub.configure_scope() as scope:
                request, error_response = old_create_request(self, *args, **kwargs)
                scope.add_event_processor(_make_asgi_request_event_processor(request))
                return (request, error_response)
        cls.create_request = sentry_patched_create_request