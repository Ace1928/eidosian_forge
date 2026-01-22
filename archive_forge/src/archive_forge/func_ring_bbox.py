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
def ring_bbox(coords):
    """Calculates and returns the bounding box of a ring.
    """
    xs, ys = zip(*coords)
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return bbox