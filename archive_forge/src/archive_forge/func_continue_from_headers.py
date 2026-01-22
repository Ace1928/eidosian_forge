import uuid
import random
from datetime import datetime, timedelta
import sentry_sdk
from sentry_sdk.consts import INSTRUMENTER
from sentry_sdk.utils import is_valid_sample_rate, logger, nanosecond_time
from sentry_sdk._compat import datetime_utcnow, utc_from_timestamp, PY2
from sentry_sdk.consts import SPANDATA
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing_utils import (
from sentry_sdk.metrics import LocalAggregator
@classmethod
def continue_from_headers(cls, headers, **kwargs):
    """
        Create a transaction with the given params (including any data pulled from
        the ``sentry-trace`` and ``baggage`` headers).

        :param headers: The dictionary with the HTTP headers to pull information from.
        """
    if cls is Span:
        logger.warning('Deprecated: use Transaction.continue_from_headers instead of Span.continue_from_headers.')
    baggage = Baggage.from_incoming_header(headers.get(BAGGAGE_HEADER_NAME))
    kwargs.update({BAGGAGE_HEADER_NAME: baggage})
    sentrytrace_kwargs = extract_sentrytrace_data(headers.get(SENTRY_TRACE_HEADER_NAME))
    if sentrytrace_kwargs is not None:
        kwargs.update(sentrytrace_kwargs)
        baggage.freeze()
    transaction = Transaction(**kwargs)
    transaction.same_process_as_parent = False
    return transaction