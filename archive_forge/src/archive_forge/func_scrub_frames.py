from sentry_sdk.utils import (
from sentry_sdk._compat import string_types
from sentry_sdk._types import TYPE_CHECKING
def scrub_frames(self, event):
    with capture_internal_exceptions():
        for frame in iter_event_frames(event):
            if 'vars' in frame:
                self.scrub_dict(frame['vars'])