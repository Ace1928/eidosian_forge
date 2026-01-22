import re
from itertools import chain
from ._base import (
from .exonerate_vulgar import _RE_VULGAR
def get_qresult_id(self, pos):
    """Return the query ID from the nearest "Query:" line."""
    handle = self._handle
    handle.seek(pos)
    sentinel = b'Query:'
    while True:
        line = handle.readline().strip()
        if line.startswith(sentinel):
            break
        if not line:
            raise StopIteration
    qid, desc = _parse_hit_or_query_line(line.decode())
    return qid