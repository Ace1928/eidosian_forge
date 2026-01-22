from __future__ import absolute_import
import json
from copy import deepcopy
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import AnnotatedValue
from sentry_sdk._compat import text_type, iteritems
from sentry_sdk._types import TYPE_CHECKING
def extract_into_event(self, event):
    client = Hub.current.client
    if client is None:
        return
    data = None
    content_length = self.content_length()
    request_info = event.get('request', {})
    if _should_send_default_pii():
        request_info['cookies'] = dict(self.cookies())
    if not request_body_within_bounds(client, content_length):
        data = AnnotatedValue.removed_because_over_size_limit()
    else:
        raw_data = None
        try:
            raw_data = self.raw_data()
        except (RawPostDataException, ValueError):
            pass
        parsed_body = self.parsed_body()
        if parsed_body is not None:
            data = parsed_body
        elif raw_data:
            data = AnnotatedValue.removed_because_raw_data()
        else:
            data = None
    if data is not None:
        request_info['data'] = data
    event['request'] = deepcopy(request_info)