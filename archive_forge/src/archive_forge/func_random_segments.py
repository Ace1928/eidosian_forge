import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def random_segments(nsegs):
    segs = []
    nbytes = 0
    for i in range(nsegs):
        seglo = np.random.randint(0, 998)
        seghi = np.random.randint(seglo + 1, 1000)
        seglen = seghi - seglo
        nbytes += seglen
        segs.append([seglo, seglen])
    return (segs, nbytes)