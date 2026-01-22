import io
import json
import mimetypes
from sentry_sdk._compat import text_type, PY2
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.session import Session
from sentry_sdk.utils import json_dumps, capture_internal_exceptions
def serialize_into(self, f):
    headers = dict(self.headers)
    bytes = self.get_bytes()
    headers['length'] = len(bytes)
    f.write(json_dumps(headers))
    f.write(b'\n')
    f.write(bytes)
    f.write(b'\n')