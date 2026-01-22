import os
import multiprocessing
from typing import TypeVar, Optional, Tuple, List
@property
def n_vocab(self) -> int:
    return self._library.rwkv_get_n_vocab(self._ctx)