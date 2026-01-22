import sys
import contextlib
from datetime import datetime, timedelta
from functools import wraps
from sentry_sdk._types import TYPE_CHECKING
def datetime_utcnow():
    return datetime.now(timezone.utc)