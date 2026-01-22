import threading
import time as mod_time
import uuid
from types import SimpleNamespace, TracebackType
from typing import Optional, Type
from redis.exceptions import LockError, LockNotOwnedError
from redis.typing import Number
def owned(self) -> bool:
    """
        Returns True if this key is locked by this lock, otherwise False.
        """
    stored_token = self.redis.get(self.name)
    if stored_token and (not isinstance(stored_token, bytes)):
        encoder = self.redis.get_encoder()
        stored_token = encoder.encode(stored_token)
    return self.local.token is not None and stored_token == self.local.token