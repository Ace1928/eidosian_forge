import sys
import contextlib
from datetime import datetime, timedelta
from functools import wraps
from sentry_sdk._types import TYPE_CHECKING
def utc_from_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc)