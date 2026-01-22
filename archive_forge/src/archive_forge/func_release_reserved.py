import contextlib
import io
import random
import struct
import time
import dns.exception
import dns.tsig
def release_reserved(self) -> None:
    """Release the reserved bytes."""
    self.max_size += self.reserved
    self.reserved = 0