from __future__ import absolute_import
import socket
from sentry_sdk import Hub
from sentry_sdk._types import MYPY
from sentry_sdk.consts import OP
from sentry_sdk.integrations import Integration
def create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    hub = Hub.current
    if hub.get_integration(SocketIntegration) is None:
        return real_create_connection(address=address, timeout=timeout, source_address=source_address)
    with hub.start_span(op=OP.SOCKET_CONNECTION, description=_get_span_description(address[0], address[1])) as span:
        span.set_data('address', address)
        span.set_data('timeout', timeout)
        span.set_data('source_address', source_address)
        return real_create_connection(address=address, timeout=timeout, source_address=source_address)