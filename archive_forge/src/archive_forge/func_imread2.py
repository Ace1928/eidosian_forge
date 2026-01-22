from __future__ import annotations
import os
from contextlib import contextmanager
import pytest
import numpy as np
from skimage.io import imsave
from dask.array.image import imread as da_imread
from dask.utils import tmpdir
def imread2(fn):
    return np.ones((2, 3, 4), dtype='i1')