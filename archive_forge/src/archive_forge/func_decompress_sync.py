import asyncio
import zlib
from concurrent.futures import Executor
from typing import Optional, cast
def decompress_sync(self, data: bytes) -> bytes:
    if hasattr(self._obj, 'decompress'):
        return cast(bytes, self._obj.decompress(data))
    return cast(bytes, self._obj.process(data))