import threading
import time as mod_time
import uuid
from types import SimpleNamespace, TracebackType
from typing import Optional, Type
from redis.exceptions import LockError, LockNotOwnedError
from redis.typing import Number
def do_acquire(self, token: str) -> bool:
    if self.timeout:
        timeout = int(self.timeout * 1000)
    else:
        timeout = None
    if self.redis.set(self.name, token, nx=True, px=timeout):
        return True
    return False