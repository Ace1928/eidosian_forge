import pytest
import random
import networkx as nx
from networkx.algorithms import approximation as approx
from networkx.algorithms import threshold
@pytest.mark.slow
def test_rng_interface():
    global progress
    for seed in [14, np.random.RandomState(14)]:
        np.random.seed(42)
        random.seed(42)
        run_all_random_functions(seed)
        progress = 0
        after_np_rv = np.random.rand()
        assert np_rv == after_np_rv
        after_py_rv = random.random()
        assert py_rv == after_py_rv