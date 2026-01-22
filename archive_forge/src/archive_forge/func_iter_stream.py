from typing import (
from urllib.parse import urlparse
def iter_stream(self) -> Iterator[bytes]:
    if not isinstance(self.stream, Iterable):
        raise RuntimeError("Attempted to stream an asynchronous response using 'for ... in response.iter_stream()'. You should use 'async for ... in response.aiter_stream()' instead.")
    if self._stream_consumed:
        raise RuntimeError("Attempted to call 'for ... in response.iter_stream()' more than once.")
    self._stream_consumed = True
    for chunk in self.stream:
        yield chunk