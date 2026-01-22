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
@contextmanager
def recursion_protection():
    """Enters recursion protection and returns the old flag."""
    old_in_metrics = _in_metrics.get()
    _in_metrics.set(True)
    try:
        yield old_in_metrics
    finally:
        _in_metrics.set(old_in_metrics)