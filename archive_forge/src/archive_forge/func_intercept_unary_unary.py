from sentry_sdk import Hub
from sentry_sdk._types import MYPY
from sentry_sdk.consts import OP
from sentry_sdk.integrations import DidNotEnable
def intercept_unary_unary(self, continuation, client_call_details, request):
    hub = Hub.current
    method = client_call_details.method
    with hub.start_span(op=OP.GRPC_CLIENT, description='unary unary call to %s' % method) as span:
        span.set_data('type', 'unary unary')
        span.set_data('method', method)
        client_call_details = self._update_client_call_details_metadata_from_hub(client_call_details, hub)
        response = continuation(client_call_details, request)
        span.set_data('code', response.code().name)
        return response