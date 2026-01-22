import typing
import warnings
from ..api import BytesBackend
from ..api import NO_VALUE
def set_serialized_multi(self, mapping):
    if not self.redis_expiration_time:
        self.writer_client.mset(mapping)
    else:
        pipe = self.writer_client.pipeline()
        for key, value in mapping.items():
            pipe.setex(key, self.redis_expiration_time, value)
        pipe.execute()