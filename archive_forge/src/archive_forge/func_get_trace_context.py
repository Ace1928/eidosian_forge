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
def get_trace_context(self):
    """
        Returns the Sentry "trace" context from the Propagation Context.
        """
    if self._propagation_context is None:
        return None
    trace_context = {'trace_id': self._propagation_context['trace_id'], 'span_id': self._propagation_context['span_id'], 'parent_span_id': self._propagation_context['parent_span_id'], 'dynamic_sampling_context': self.get_dynamic_sampling_context()}
    return trace_context