from contextlib import suppress
from functools import partial
from .encode import Encode
def bytes_concat(L):
    return b''.join(L)