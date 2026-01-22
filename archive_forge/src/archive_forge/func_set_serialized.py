import typing
import warnings
from ..api import BytesBackend
from ..api import NO_VALUE
def set_serialized(self, key, value):
    if self.redis_expiration_time:
        self.writer_client.setex(key, self.redis_expiration_time, value)
    else:
        self.writer_client.set(key, value)