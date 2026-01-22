import numpy as np
from scipy import stats
from tune._utils import (
def test_uniform_to_choice():
    assert 'a' == uniform_to_choice(0.5, ['a'])
    np.random.seed(0)
    values = np.random.uniform(0, 1.0, 1000)
    res = uniform_to_choice(values, ['a', 'b', 'c', 'd'])
    for c in ['a', 'b', 'c', 'd']:
        assert sum((1 if v == c else 0 for v in res)) >= 230