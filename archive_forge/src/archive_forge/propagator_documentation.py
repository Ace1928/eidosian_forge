from opentelemetry import trace  # type: ignore
from opentelemetry.context import (  # type: ignore
from opentelemetry.propagators.textmap import (  # type: ignore
from opentelemetry.trace import (  # type: ignore
from sentry_sdk.integrations.opentelemetry.consts import (
from sentry_sdk.integrations.opentelemetry.span_processor import (
from sentry_sdk.tracing import (
from sentry_sdk.tracing_utils import Baggage, extract_sentrytrace_data
from sentry_sdk._types import TYPE_CHECKING

    Propagates tracing headers for Sentry's tracing system in a way OTel understands.
    