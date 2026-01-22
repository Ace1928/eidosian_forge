import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def snap_edges(edges, x_tolerance=DEFAULT_SNAP_TOLERANCE, y_tolerance=DEFAULT_SNAP_TOLERANCE):
    """
    Given a list of edges, snap any within `tolerance` pixels of one another
    to their positional average.
    """
    by_orientation = {'v': [], 'h': []}
    for e in edges:
        by_orientation[e['orientation']].append(e)
    snapped_v = snap_objects(by_orientation['v'], 'x0', x_tolerance)
    snapped_h = snap_objects(by_orientation['h'], 'top', y_tolerance)
    return snapped_v + snapped_h