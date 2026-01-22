import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def make_line(p, p1, p2, clip):
    """Given 2 points, make a line dictionary for table detection."""
    if not is_parallel(p1, p2):
        return {}
    x0 = min(p1.x, p2.x)
    x1 = max(p1.x, p2.x)
    y0 = min(p1.y, p2.y)
    y1 = max(p1.y, p2.y)
    if x0 > clip.x1 or x1 < clip.x0 or y0 > clip.y1 or (y1 < clip.y0):
        return {}
    if x0 < clip.x0:
        x0 = clip.x0
    if x1 > clip.x1:
        x1 = clip.x1
    if y0 < clip.y0:
        y0 = clip.y0
    if y1 > clip.y1:
        y1 = clip.y1
    width = x1 - x0
    height = y1 - y0
    if width == height == 0:
        return {}
    line_dict = {'x0': x0, 'y0': page_height - y0, 'x1': x1, 'y1': page_height - y1, 'width': width, 'height': height, 'pts': [(x0, y0), (x1, y1)], 'linewidth': p['width'], 'stroke': True, 'fill': False, 'evenodd': False, 'stroking_color': p['color'] if p['color'] else p['fill'], 'non_stroking_color': None, 'object_type': 'line', 'page_number': page_number, 'stroking_pattern': None, 'non_stroking_pattern': None, 'top': y0, 'bottom': y1, 'doctop': y0 + doctop_basis}
    return line_dict