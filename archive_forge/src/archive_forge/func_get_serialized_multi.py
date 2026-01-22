import typing
import warnings
from ..api import BytesBackend
from ..api import NO_VALUE
def get_serialized_multi(self, keys):
    if not keys:
        return []
    values = self.reader_client.mget(keys)
    return [v if v is not None else NO_VALUE for v in values]