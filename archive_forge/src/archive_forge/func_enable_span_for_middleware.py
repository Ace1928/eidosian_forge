from typing import TYPE_CHECKING
from pydantic import BaseModel  # type: ignore
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.utils import event_from_exception, transaction_from_function
def enable_span_for_middleware(middleware: 'Middleware') -> 'Middleware':
    if not hasattr(middleware, '__call__') or middleware is SentryStarliteASGIMiddleware:
        return middleware
    if isinstance(middleware, DefineMiddleware):
        old_call: 'ASGIApp' = middleware.middleware.__call__
    else:
        old_call = middleware.__call__

    async def _create_span_call(self: 'MiddlewareProtocol', scope: 'Scope', receive: 'Receive', send: 'Send') -> None:
        hub = Hub.current
        integration = hub.get_integration(StarliteIntegration)
        if integration is not None:
            middleware_name = self.__class__.__name__
            with hub.start_span(op=OP.MIDDLEWARE_STARLITE, description=middleware_name) as middleware_span:
                middleware_span.set_tag('starlite.middleware_name', middleware_name)

                async def _sentry_receive(*args: 'Any', **kwargs: 'Any') -> 'Union[HTTPReceiveMessage, WebSocketReceiveMessage]':
                    hub = Hub.current
                    with hub.start_span(op=OP.MIDDLEWARE_STARLITE_RECEIVE, description=getattr(receive, '__qualname__', str(receive))) as span:
                        span.set_tag('starlite.middleware_name', middleware_name)
                        return await receive(*args, **kwargs)
                receive_name = getattr(receive, '__name__', str(receive))
                receive_patched = receive_name == '_sentry_receive'
                new_receive = _sentry_receive if not receive_patched else receive

                async def _sentry_send(message: 'Message') -> None:
                    hub = Hub.current
                    with hub.start_span(op=OP.MIDDLEWARE_STARLITE_SEND, description=getattr(send, '__qualname__', str(send))) as span:
                        span.set_tag('starlite.middleware_name', middleware_name)
                        return await send(message)
                send_name = getattr(send, '__name__', str(send))
                send_patched = send_name == '_sentry_send'
                new_send = _sentry_send if not send_patched else send
                return await old_call(self, scope, new_receive, new_send)
        else:
            return await old_call(self, scope, receive, send)
    not_yet_patched = old_call.__name__ not in ['_create_span_call']
    if not_yet_patched:
        if isinstance(middleware, DefineMiddleware):
            middleware.middleware.__call__ = _create_span_call
        else:
            middleware.__call__ = _create_span_call
    return middleware