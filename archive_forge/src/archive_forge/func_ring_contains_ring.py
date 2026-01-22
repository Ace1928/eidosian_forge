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
def ring_contains_ring(coords1, coords2):
    """Returns True if all vertexes in coords2 are fully inside coords1.
    """
    return all((ring_contains_point(coords1, p2) for p2 in coords2))