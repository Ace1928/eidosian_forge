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
def update_from_kwargs(self, user=None, level=None, extras=None, contexts=None, tags=None, fingerprint=None):
    """Update the scope's attributes."""
    if level is not None:
        self._level = level
    if user is not None:
        self._user = user
    if extras is not None:
        self._extras.update(extras)
    if contexts is not None:
        self._contexts.update(contexts)
    if tags is not None:
        self._tags.update(tags)
    if fingerprint is not None:
        self._fingerprint = fingerprint