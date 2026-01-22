from opentelemetry import trace  # type: ignore
from opentelemetry.context import (  # type: ignore
from opentelemetry.propagators.textmap import (  # type: ignore
from opentelemetry.trace import (  # type: ignore
from sentry_sdk.integrations.opentelemetry.consts import (
from sentry_sdk.integrations.opentelemetry.span_processor import (
from sentry_sdk.tracing import (
from sentry_sdk.tracing_utils import Baggage, extract_sentrytrace_data
from sentry_sdk._types import TYPE_CHECKING
def inject(self, carrier, context=None, setter=default_setter):
    if context is None:
        context = get_current()
    current_span = trace.get_current_span(context)
    current_span_context = current_span.get_span_context()
    if not current_span_context.is_valid:
        return
    span_id = trace.format_span_id(current_span_context.span_id)
    span_map = SentrySpanProcessor().otel_span_map
    sentry_span = span_map.get(span_id, None)
    if not sentry_span:
        return
    setter.set(carrier, SENTRY_TRACE_HEADER_NAME, sentry_span.to_traceparent())
    if sentry_span.containing_transaction:
        baggage = sentry_span.containing_transaction.get_baggage()
        if baggage:
            setter.set(carrier, BAGGAGE_HEADER_NAME, baggage.serialize())