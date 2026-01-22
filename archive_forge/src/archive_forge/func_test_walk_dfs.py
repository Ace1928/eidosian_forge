import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_walk_dfs():
    relu = Relu(5)
    relu2 = Relu(5)
    inner_chain = chain(relu, relu2)
    chained = chain(inner_chain, inner_chain)
    assert list(chained.walk(order='dfs_pre')) == [chained, inner_chain, relu, relu2]
    assert list(chained.walk(order='dfs_post')) == [relu, relu2, inner_chain, chained]