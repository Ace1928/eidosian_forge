import io
import os
import sys
from pyasn1 import error
from pyasn1.type import univ
@markedPosition.setter
def markedPosition(self, value):
    self._markedPosition = value
    if self._cache.tell() > io.DEFAULT_BUFFER_SIZE:
        self._cache = io.BytesIO(self._cache.read())
        self._markedPosition = 0