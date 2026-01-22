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
def update_from_scope(self, scope):
    """Update the scope with another scope's data."""
    if scope._level is not None:
        self._level = scope._level
    if scope._fingerprint is not None:
        self._fingerprint = scope._fingerprint
    if scope._transaction is not None:
        self._transaction = scope._transaction
    if scope._transaction_info is not None:
        self._transaction_info.update(scope._transaction_info)
    if scope._user is not None:
        self._user = scope._user
    if scope._tags:
        self._tags.update(scope._tags)
    if scope._contexts:
        self._contexts.update(scope._contexts)
    if scope._extras:
        self._extras.update(scope._extras)
    if scope._breadcrumbs:
        self._breadcrumbs.extend(scope._breadcrumbs)
    if scope._span:
        self._span = scope._span
    if scope._attachments:
        self._attachments.extend(scope._attachments)
    if scope._profile:
        self._profile = scope._profile
    if scope._propagation_context:
        self._propagation_context = scope._propagation_context