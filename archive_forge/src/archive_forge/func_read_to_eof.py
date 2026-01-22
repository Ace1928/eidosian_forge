from __future__ import annotations
from typing import Generator
def read_to_eof(self, m: int) -> Generator[None, None, bytes]:
    """
        Read all bytes from the stream.

        This is a generator-based coroutine.

        Args:
            m: maximum number bytes to read; this is a security limit.

        Raises:
            RuntimeError: if the stream ends in more than ``m`` bytes.

        """
    while not self.eof:
        p = len(self.buffer)
        if p > m:
            raise RuntimeError(f'read {p} bytes, expected no more than {m} bytes')
        yield
    r = self.buffer[:]
    del self.buffer[:]
    return r