import sys
import contextlib
from datetime import datetime, timedelta
from functools import wraps
from sentry_sdk._types import TYPE_CHECKING
def duration_in_milliseconds(delta):
    return delta / timedelta(milliseconds=1)