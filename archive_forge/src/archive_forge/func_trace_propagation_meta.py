from copy import copy
from collections import deque
from itertools import chain
import os
import sys
import uuid
from sentry_sdk.attachments import Attachment
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk.consts import FALSE_VALUES, INSTRUMENTER
from sentry_sdk._functools import wraps
from sentry_sdk.profiler import Profile
from sentry_sdk.session import Session
from sentry_sdk.tracing_utils import (
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def trace_propagation_meta(self, *args, **kwargs):
    """
        Return meta tags which should be injected into HTML templates
        to allow propagation of trace information.
        """
    span = kwargs.pop('span', None)
    if span is not None:
        logger.warning('The parameter `span` in trace_propagation_meta() is deprecated and will be removed in the future.')
    client = kwargs.pop('client', None)
    meta = ''
    sentry_trace = self.get_traceparent(client=client)
    if sentry_trace is not None:
        meta += '<meta name="%s" content="%s">' % (SENTRY_TRACE_HEADER_NAME, sentry_trace)
    baggage = self.get_baggage(client=client)
    if baggage is not None:
        meta += '<meta name="%s" content="%s">' % (BAGGAGE_HEADER_NAME, baggage.serialize())
    return meta