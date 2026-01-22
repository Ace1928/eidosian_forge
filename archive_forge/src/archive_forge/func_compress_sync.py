import asyncio
import zlib
from concurrent.futures import Executor
from typing import Optional, cast
def compress_sync(self, data: bytes) -> bytes:
    return self._compressor.compress(data)