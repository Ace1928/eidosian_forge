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
def generate_propagation_context(self, incoming_data=None):
    """
        Makes sure `_propagation_context` is set.
        If there is `incoming_data` overwrite existing `_propagation_context`.
        if there is no `incoming_data` create new `_propagation_context`, but do NOT overwrite if already existing.
        """
    if incoming_data:
        context = self._extract_propagation_context(incoming_data)
        if context is not None:
            self._propagation_context = context
            logger.debug('[Tracing] Extracted propagation context from incoming data: %s', self._propagation_context)
    if self._propagation_context is None:
        self.set_new_propagation_context()