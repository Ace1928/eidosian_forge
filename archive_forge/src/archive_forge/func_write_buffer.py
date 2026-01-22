import json
from typing import IO, List, Tuple, Any
from .parser import Parser
from .symbols import (
def write_buffer(self):
    json_data = '\n'.join([json.dumps(record) for record in self._records])
    self._fo.write(json_data)