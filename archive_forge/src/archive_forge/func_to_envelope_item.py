import os
import mimetypes
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.envelope import Item, PayloadRef
def to_envelope_item(self):
    """Returns an envelope item for this attachment."""
    payload = None
    if self.bytes is not None:
        if callable(self.bytes):
            payload = self.bytes()
        else:
            payload = self.bytes
    else:
        payload = PayloadRef(path=self.path)
    return Item(payload=payload, type='attachment', content_type=self.content_type, filename=self.filename)