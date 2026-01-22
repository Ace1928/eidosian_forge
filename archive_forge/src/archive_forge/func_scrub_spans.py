from sentry_sdk.utils import (
from sentry_sdk._compat import string_types
from sentry_sdk._types import TYPE_CHECKING
def scrub_spans(self, event):
    with capture_internal_exceptions():
        if 'spans' in event:
            for span in event['spans']:
                if 'data' in span:
                    self.scrub_dict(span['data'])