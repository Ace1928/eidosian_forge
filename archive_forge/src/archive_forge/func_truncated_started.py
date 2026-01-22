import uuid
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import format_timestamp
@property
def truncated_started(self):
    return _minute_trunc(self.started)