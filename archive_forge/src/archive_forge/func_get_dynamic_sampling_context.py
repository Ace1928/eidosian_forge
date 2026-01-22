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
def get_dynamic_sampling_context(self):
    """
        Returns the Dynamic Sampling Context from the Propagation Context.
        If not existing, creates a new one.
        """
    if self._propagation_context is None:
        return None
    baggage = self.get_baggage()
    if baggage is not None:
        self._propagation_context['dynamic_sampling_context'] = baggage.dynamic_sampling_context()
    return self._propagation_context['dynamic_sampling_context']