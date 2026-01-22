from __future__ import print_function
import io
import gzip
import time
from datetime import timedelta
from collections import defaultdict
import urllib3
import certifi
from sentry_sdk.utils import Dsn, logger, capture_internal_exceptions, json_dumps
from sentry_sdk.worker import BackgroundWorker
from sentry_sdk.envelope import Envelope, Item, PayloadRef
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk._types import TYPE_CHECKING
def record_lost_event(self, reason, data_category=None, item=None):
    if not self.options['send_client_reports']:
        return
    quantity = 1
    if item is not None:
        data_category = item.data_category
        if data_category == 'attachment':
            quantity = len(item.get_bytes()) or 1
    elif data_category is None:
        raise TypeError('data category not provided')
    self._discarded_events[data_category, reason] += quantity