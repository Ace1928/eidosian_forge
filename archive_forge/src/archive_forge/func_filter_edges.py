import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def filter_edges(edges, orientation=None, edge_type=None, min_length=1) -> list:
    if orientation not in ('v', 'h', None):
        raise ValueError("Orientation must be 'v' or 'h'")

    def test(e) -> bool:
        dim = 'height' if e['orientation'] == 'v' else 'width'
        et_correct = e['object_type'] == edge_type if edge_type is not None else True
        orient_correct = orientation is None or e['orientation'] == orientation
        return bool(et_correct and orient_correct and (e[dim] >= min_length))
    return list(filter(test, edges))