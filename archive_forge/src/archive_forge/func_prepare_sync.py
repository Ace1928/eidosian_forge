import asyncio
import functools
import queue
import threading
import time
from typing import (
@functools.wraps(prepare_async)
def prepare_sync(self, file_spec: 'CreateArtifactFileSpecInput') -> 'queue.Queue[ResponsePrepare]':
    response_queue: queue.Queue[ResponsePrepare] = queue.Queue()
    self._request_queue.put(RequestPrepare(file_spec, response_queue))
    return response_queue