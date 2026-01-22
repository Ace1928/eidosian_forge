from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def ring_sample(coords, ccw=False):
    """Return a sample point guaranteed to be within a ring, by efficiently
    finding the first centroid of a coordinate triplet whose orientation
    matches the orientation of the ring and passes the point-in-ring test.
    The orientation of the ring is assumed to be clockwise, unless ccw
    (counter-clockwise) is set to True. 
    """
    triplet = []

    def itercoords():
        for p in coords:
            yield p
        yield coords[1]
    for p in itercoords():
        if p not in triplet:
            triplet.append(p)
        if len(triplet) == 3:
            is_straight_line = (triplet[0][1] - triplet[1][1]) * (triplet[0][0] - triplet[2][0]) == (triplet[0][1] - triplet[2][1]) * (triplet[0][0] - triplet[1][0])
            if not is_straight_line:
                closed_triplet = triplet + [triplet[0]]
                triplet_ccw = not is_cw(closed_triplet)
                if ccw == triplet_ccw:
                    xs, ys = zip(*triplet)
                    xmean, ymean = (sum(xs) / 3.0, sum(ys) / 3.0)
                    if ring_contains_point(coords, (xmean, ymean)):
                        return (xmean, ymean)
            triplet.pop(0)
    else:
        raise Exception('Unexpected error: Unable to find a ring sample point.')