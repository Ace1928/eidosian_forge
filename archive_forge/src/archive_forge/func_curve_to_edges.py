import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def curve_to_edges(curve) -> list:
    point_pairs = zip(curve['pts'], curve['pts'][1:])
    return [{'object_type': 'curve_edge', 'x0': min(p0[0], p1[0]), 'x1': max(p0[0], p1[0]), 'top': min(p0[1], p1[1]), 'doctop': min(p0[1], p1[1]) + (curve['doctop'] - curve['top']), 'bottom': max(p0[1], p1[1]), 'width': abs(p0[0] - p1[0]), 'height': abs(p0[1] - p1[1]), 'orientation': 'v' if p0[0] == p1[0] else 'h' if p0[1] == p1[1] else None} for p0, p1 in point_pairs]