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
def capture_envelope(self, envelope):
    hub = self.hub_cls.current

    def send_envelope_wrapper():
        with hub:
            with capture_internal_exceptions():
                self._send_envelope(envelope)
                self._flush_client_reports()
    if not self._worker.submit(send_envelope_wrapper):
        self.on_dropped_event('full_queue')
        for item in envelope.items:
            self.record_lost_event('queue_overflow', item=item)