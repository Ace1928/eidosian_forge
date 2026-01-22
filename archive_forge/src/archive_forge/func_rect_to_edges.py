import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def rect_to_edges(rect) -> list:
    top, bottom, left, right = [dict(rect) for x in range(4)]
    top.update({'object_type': 'rect_edge', 'height': 0, 'y0': rect['y1'], 'bottom': rect['top'], 'orientation': 'h'})
    bottom.update({'object_type': 'rect_edge', 'height': 0, 'y1': rect['y0'], 'top': rect['top'] + rect['height'], 'doctop': rect['doctop'] + rect['height'], 'orientation': 'h'})
    left.update({'object_type': 'rect_edge', 'width': 0, 'x1': rect['x0'], 'orientation': 'v'})
    right.update({'object_type': 'rect_edge', 'width': 0, 'x0': rect['x1'], 'orientation': 'v'})
    return [top, bottom, left, right]