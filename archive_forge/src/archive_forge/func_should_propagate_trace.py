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
def should_propagate_trace(hub, url):
    """
    Returns True if url matches trace_propagation_targets configured in the given hub. Otherwise, returns False.
    """
    client = hub.client
    trace_propagation_targets = client.options['trace_propagation_targets']
    if is_sentry_url(hub, url):
        return False
    return match_regex_list(url, trace_propagation_targets, substring_matching=True)