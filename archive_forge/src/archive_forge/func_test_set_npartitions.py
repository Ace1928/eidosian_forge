import numpy as np
import pytest
import modin.pandas as pd
from modin.config import NPartitions
@pytest.mark.parametrize('num_partitions', [2, 4, 6, 8, 10])
def test_set_npartitions(num_partitions):
    NPartitions.put(num_partitions)
    data = np.random.randint(0, 100, size=(2 ** 16, 2 ** 8))
    df = pd.DataFrame(data)
    part_shape = df._query_compiler._modin_frame._partitions.shape
    assert part_shape[0] == num_partitions and part_shape[1] == min(num_partitions, 8)