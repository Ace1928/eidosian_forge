import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def edges_to_intersections(edges, x_tolerance=1, y_tolerance=1) -> dict:
    """
    Given a list of edges, return the points at which they intersect
    within `tolerance` pixels.
    """
    intersections = {}
    v_edges, h_edges = [list(filter(lambda x: x['orientation'] == o, edges)) for o in ('v', 'h')]
    for v in sorted(v_edges, key=itemgetter('x0', 'top')):
        for h in sorted(h_edges, key=itemgetter('top', 'x0')):
            if v['top'] <= h['top'] + y_tolerance and v['bottom'] >= h['top'] - y_tolerance and (v['x0'] >= h['x0'] - x_tolerance) and (v['x0'] <= h['x1'] + x_tolerance):
                vertex = (v['x0'], h['top'])
                if vertex not in intersections:
                    intersections[vertex] = {'v': [], 'h': []}
                intersections[vertex]['v'].append(v)
                intersections[vertex]['h'].append(h)
    return intersections