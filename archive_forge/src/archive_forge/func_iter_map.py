import json
from typing import IO, Any, Tuple, List
from .parser import Parser
from .symbols import (
def iter_map(self):
    while len(self._current) > 0:
        self._push()
        for key in self._current:
            break
        yield
        self._pop()
        del self._current[key]