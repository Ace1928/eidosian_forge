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
def set_transaction_name(self, name, source=None):
    """Set the transaction name and optionally the transaction source."""
    self._transaction = name
    if self._span and self._span.containing_transaction:
        self._span.containing_transaction.name = name
        if source:
            self._span.containing_transaction.source = source
    if source:
        self._transaction_info['source'] = source