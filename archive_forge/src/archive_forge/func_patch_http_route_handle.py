from typing import TYPE_CHECKING
from pydantic import BaseModel  # type: ignore
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.utils import event_from_exception, transaction_from_function
def patch_http_route_handle() -> None:
    old_handle = HTTPRoute.handle

    async def handle_wrapper(self: 'HTTPRoute', scope: 'HTTPScope', receive: 'Receive', send: 'Send') -> None:
        hub = Hub.current
        integration: StarliteIntegration = hub.get_integration(StarliteIntegration)
        if integration is None:
            return await old_handle(self, scope, receive, send)
        with hub.configure_scope() as sentry_scope:
            request: 'Request[Any, Any]' = scope['app'].request_class(scope=scope, receive=receive, send=send)
            extracted_request_data = ConnectionDataExtractor(parse_body=True, parse_query=True)(request)
            body = extracted_request_data.pop('body')
            request_data = await body

            def event_processor(event: 'Event', _: 'Dict[str, Any]') -> 'Event':
                route_handler = scope.get('route_handler')
                request_info = event.get('request', {})
                request_info['content_length'] = len(scope.get('_body', b''))
                if _should_send_default_pii():
                    request_info['cookies'] = extracted_request_data['cookies']
                if request_data is not None:
                    request_info['data'] = request_data
                func = None
                if route_handler.name is not None:
                    tx_name = route_handler.name
                elif isinstance(route_handler.fn, Ref):
                    func = route_handler.fn.value
                else:
                    func = route_handler.fn
                if func is not None:
                    tx_name = transaction_from_function(func)
                tx_info = {'source': SOURCE_FOR_STYLE['endpoint']}
                if not tx_name:
                    tx_name = _DEFAULT_TRANSACTION_NAME
                    tx_info = {'source': TRANSACTION_SOURCE_ROUTE}
                event.update({'request': request_info, 'transaction': tx_name, 'transaction_info': tx_info})
                return event
            sentry_scope._name = StarliteIntegration.identifier
            sentry_scope.add_event_processor(event_processor)
            return await old_handle(self, scope, receive, send)
    HTTPRoute.handle = handle_wrapper