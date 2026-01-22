import contextlib
import os
import re
import sys
import sentry_sdk
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.utils import (
from sentry_sdk._compat import PY2, duration_in_milliseconds, iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing import LOW_QUALITY_TRANSACTION_SOURCES
def has_tracing_enabled(options):
    """
    Returns True if either traces_sample_rate or traces_sampler is
    defined and enable_tracing is set and not false.
    """
    if options is None:
        return False
    return bool(options.get('enable_tracing') is not False and (options.get('traces_sample_rate') is not None or options.get('traces_sampler') is not None))