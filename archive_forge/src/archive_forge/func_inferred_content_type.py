import io
import json
import mimetypes
from sentry_sdk._compat import text_type, PY2
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.session import Session
from sentry_sdk.utils import json_dumps, capture_internal_exceptions
@property
def inferred_content_type(self):
    if self.json is not None:
        return 'application/json'
    elif self.path is not None:
        path = self.path
        if isinstance(path, bytes):
            path = path.decode('utf-8', 'replace')
        ty = mimetypes.guess_type(path)[0]
        if ty:
            return ty
    return 'application/octet-stream'