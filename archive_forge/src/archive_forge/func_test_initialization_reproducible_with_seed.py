import random
import numpy as np
import pytest
import networkx as nx
import cirq
import cirq.contrib.routing as ccr
@pytest.mark.parametrize('seed', [random.randint(0, 2 ** 32) for _ in range(10)])
def test_initialization_reproducible_with_seed(seed):
    wrappers = (lambda s: s, np.random.RandomState)
    mappings = [get_seeded_initial_mapping(seed, wrapper(seed)) for wrapper in wrappers for _ in range(5)]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*mappings)