import weakref
import contextlib
from inspect import iscoroutinefunction
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
def tornado_processor(event, hint):
    handler = weak_handler()
    if handler is None:
        return event
    request = handler.request
    with capture_internal_exceptions():
        method = getattr(handler, handler.request.method.lower())
        event['transaction'] = transaction_from_function(method) or ''
        event['transaction_info'] = {'source': TRANSACTION_SOURCE_COMPONENT}
    with capture_internal_exceptions():
        extractor = TornadoRequestExtractor(request)
        extractor.extract_into_event(event)
        request_info = event['request']
        request_info['url'] = '%s://%s%s' % (request.protocol, request.host, request.path)
        request_info['query_string'] = request.query
        request_info['method'] = request.method
        request_info['env'] = {'REMOTE_ADDR': request.remote_ip}
        request_info['headers'] = _filter_headers(dict(request.headers))
    with capture_internal_exceptions():
        if handler.current_user and _should_send_default_pii():
            event.setdefault('user', {}).setdefault('is_authenticated', True)
    return event