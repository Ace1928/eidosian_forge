import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface, PythonRandomViaNumpyBits
def test_random_state_np_random_Generator(self):
    np.random.seed(42)
    np_rv = np.random.random()
    np.random.seed(42)
    seed = 1
    rng = np.random.default_rng(seed)
    rval = self.instantiate_np_random_state(rng)
    rval_expected = np.random.default_rng(seed).random()
    assert rval == rval_expected
    rval = self.instantiate_py_random_state(rng)
    rval_expected = np.random.default_rng(seed).random(size=2)[1]
    assert rval == rval_expected
    assert np_rv == np.random.random()