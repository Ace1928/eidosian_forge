import sys
import weakref
from sentry_sdk.api import continue_trace
from sentry_sdk._compat import reraise
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.sessions import auto_session_tracking
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.tracing import (
from sentry_sdk.tracing_utils import should_propagate_trace
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def create_trace_config():

    async def on_request_start(session, trace_config_ctx, params):
        hub = Hub.current
        if hub.get_integration(AioHttpIntegration) is None:
            return
        method = params.method.upper()
        parsed_url = None
        with capture_internal_exceptions():
            parsed_url = parse_url(str(params.url), sanitize=False)
        span = hub.start_span(op=OP.HTTP_CLIENT, description='%s %s' % (method, parsed_url.url if parsed_url else SENSITIVE_DATA_SUBSTITUTE))
        span.set_data(SPANDATA.HTTP_METHOD, method)
        if parsed_url is not None:
            span.set_data('url', parsed_url.url)
            span.set_data(SPANDATA.HTTP_QUERY, parsed_url.query)
            span.set_data(SPANDATA.HTTP_FRAGMENT, parsed_url.fragment)
        if should_propagate_trace(hub, str(params.url)):
            for key, value in hub.iter_trace_propagation_headers(span):
                logger.debug('[Tracing] Adding `{key}` header {value} to outgoing request to {url}.'.format(key=key, value=value, url=params.url))
                if key == BAGGAGE_HEADER_NAME and params.headers.get(BAGGAGE_HEADER_NAME):
                    params.headers[key] += ',' + value
                else:
                    params.headers[key] = value
        trace_config_ctx.span = span

    async def on_request_end(session, trace_config_ctx, params):
        if trace_config_ctx.span is None:
            return
        span = trace_config_ctx.span
        span.set_http_status(int(params.response.status))
        span.set_data('reason', params.response.reason)
        span.finish()
    trace_config = TraceConfig()
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_end.append(on_request_end)
    return trace_config