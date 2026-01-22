from __future__ import annotations
import random
import time
from operator import add
import pytest
import dask
import dask.bag as db
from dask import delayed
from dask.base import clone_key
from dask.blockwise import Blockwise
from dask.graph_manipulation import bind, checkpoint, chunks, clone, wait_on
from dask.highlevelgraph import HighLevelGraph
from dask.tests.test_base import Tuple
from dask.utils_test import import_or_none
def test_split_every_invalid():
    t = Tuple({'a': 1, 'b': 2}, ['a', 'b'])
    with pytest.raises(ValueError):
        checkpoint(t, split_every=1)
    with pytest.raises(ValueError):
        checkpoint(t, split_every=1.9)
    with pytest.raises(ValueError):
        checkpoint(t, split_every=0)
    with pytest.raises(ValueError):
        checkpoint(t, split_every=-2)
    with pytest.raises(TypeError):
        checkpoint(t, split_every={0: 2})