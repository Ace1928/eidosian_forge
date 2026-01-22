import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def datasource1_direct():
    """A short dataset"""
    schema = datasource1_schema()

    class Generator:

        def __init__(self):
            self.n = 3

        def __call__(self, ctx):
            if self.n == 0:
                batch = _record_batch_from_iters(schema, [], [])
            else:
                self.n -= 1
                batch = _record_batch_for_range(schema, self.n)
            return batch.to_struct_array()
    return lambda ctx: Generator()