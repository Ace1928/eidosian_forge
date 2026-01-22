import io
import json
import mimetypes
from sentry_sdk._compat import text_type, PY2
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.session import Session
from sentry_sdk.utils import json_dumps, capture_internal_exceptions
def get_event(self):
    """
        Returns an error event if there is one.
        """
    if self.type == 'event' and self.payload.json is not None:
        return self.payload.json
    return None