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
def set_new_propagation_context(self):
    """
        Creates a new propagation context and sets it as `_propagation_context`. Overwriting existing one.
        """
    self._propagation_context = self._create_new_propagation_context()
    logger.debug('[Tracing] Create new propagation context: %s', self._propagation_context)