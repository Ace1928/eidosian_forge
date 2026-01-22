from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def get_str_bbox_and_descent(self, s):
    """Return the string bounding box and the maximal descent."""
    if not len(s):
        return (0, 0, 0, 0, 0)
    total_width = 0
    namelast = None
    miny = 1000000000.0
    maxy = 0
    left = 0
    if not isinstance(s, str):
        s = _to_str(s)
    for c in s:
        if c == '\n':
            continue
        name = uni2type1.get(ord(c), f'uni{ord(c):04X}')
        try:
            wx, _, bbox = self._metrics_by_name[name]
        except KeyError:
            name = 'question'
            wx, _, bbox = self._metrics_by_name[name]
        total_width += wx + self._kern.get((namelast, name), 0)
        l, b, w, h = bbox
        left = min(left, l)
        miny = min(miny, b)
        maxy = max(maxy, b + h)
        namelast = name
    return (left, miny, total_width, maxy - miny, -miny)