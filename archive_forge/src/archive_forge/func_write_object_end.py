import json
from typing import IO, List, Tuple, Any
from .parser import Parser
from .symbols import (
def write_object_end(self):
    self._pop()