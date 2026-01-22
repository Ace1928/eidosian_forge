import asyncio
import functools
import queue
import threading
import time
from typing import (
def prepare_async(self, file_spec: 'CreateArtifactFileSpecInput') -> 'asyncio.Future[ResponsePrepare]':
    """Request the backend to prepare a file for upload."""
    response: asyncio.Future[ResponsePrepare] = asyncio.Future()
    self._request_queue.put(RequestPrepare(file_spec, (asyncio.get_event_loop(), response)))
    return response