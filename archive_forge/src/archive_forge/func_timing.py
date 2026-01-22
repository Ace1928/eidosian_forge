import io
import os
import random
import re
import sys
import threading
import time
import zlib
from contextlib import contextmanager
from datetime import datetime
from functools import wraps, partial
import sentry_sdk
from sentry_sdk._compat import text_type, utc_from_timestamp, iteritems
from sentry_sdk.utils import (
from sentry_sdk.envelope import Envelope, Item
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
def timing(key, value=None, unit='second', tags=None, timestamp=None, stacklevel=0):
    """Emits a distribution with the time it takes to run the given code block.

    This method supports three forms of invocation:

    - when a `value` is provided, it functions similar to `distribution` but with
    - it can be used as a context manager
    - it can be used as a decorator
    """
    if value is not None:
        aggregator, local_aggregator, tags = _get_aggregator_and_update_tags(key, tags)
        if aggregator is not None:
            aggregator.add('d', key, value, unit, tags, timestamp, local_aggregator, stacklevel)
    return _Timing(key, tags, timestamp, value, unit, stacklevel)