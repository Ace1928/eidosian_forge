import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_walk_bfs_post_order_fails():
    relu = Relu(5)
    with pytest.raises(ValueError, match='Invalid order'):
        relu.walk(order='dfs_post_order')