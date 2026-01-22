from __future__ import absolute_import
import math, sys
def prange(self, start=0, stop=None, step=1, nogil=False, schedule=None, chunksize=None, num_threads=None):
    if stop is None:
        stop = start
        start = 0
    return range(start, stop, step)