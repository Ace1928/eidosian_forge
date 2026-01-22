from __future__ import annotations
from .. import http
def range_for_length(self, length):
    """If the range is for bytes, the length is not None and there is
        exactly one range and it is satisfiable it returns a ``(start, stop)``
        tuple, otherwise `None`.
        """
    if self.units != 'bytes' or length is None or len(self.ranges) != 1:
        return None
    start, end = self.ranges[0]
    if end is None:
        end = length
        if start < 0:
            start += length
    if http.is_byte_range_valid(start, end, length):
        return (start, min(end, length))
    return None