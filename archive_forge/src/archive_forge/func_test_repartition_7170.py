import numpy as np
import pytest
import modin.pandas as pd
from modin.config import NPartitions, context
def test_repartition_7170():
    with context(MinPartitionSize=102, NPartitions=5):
        df = pd.DataFrame(np.random.rand(10000, 100))
        _ = df._repartition(axis=1).to_numpy()