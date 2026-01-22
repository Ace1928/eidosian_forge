import os
import random
import time
from ._compat import long, binary_type
def random_8(self):
    self.lock.acquire()
    try:
        self._maybe_seed()
        if self.digest is None or self.next_byte == self.hash_len:
            self.hash.update(binary_type(self.pool))
            self.digest = bytearray(self.hash.digest())
            self.stir(self.digest, True)
            self.next_byte = 0
        value = self.digest[self.next_byte]
        self.next_byte += 1
    finally:
        self.lock.release()
    return value