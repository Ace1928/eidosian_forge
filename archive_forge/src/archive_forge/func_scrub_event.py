from sentry_sdk.utils import (
from sentry_sdk._compat import string_types
from sentry_sdk._types import TYPE_CHECKING
def scrub_event(self, event):
    self.scrub_request(event)
    self.scrub_extra(event)
    self.scrub_user(event)
    self.scrub_breadcrumbs(event)
    self.scrub_frames(event)
    self.scrub_spans(event)