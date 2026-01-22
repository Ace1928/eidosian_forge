import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_random_state_int(self):
    np.random.seed(42)
    np_rv = np.random.random_sample()
    random.seed(42)
    py_rv = random.random()
    np.random.seed(42)
    seed = 1
    rval = self.instantiate_np_random_state(seed)
    rval_expected = np.random.RandomState(seed).rand()
    assert rval, rval_expected
    assert np_rv == np.random.random_sample()
    random.seed(42)
    rval = self.instantiate_py_random_state(seed)
    rval_expected = random.Random(seed).random()
    assert rval, rval_expected
    assert py_rv == random.random()