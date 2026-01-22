import sys
import sysconfig
import pytest
import pyarrow as pa
import numpy as np
def make_recordbatch(length):
    schema = pa.schema([pa.field('f0', pa.int16()), pa.field('f1', pa.int16())])
    a0 = pa.array(np.random.randint(0, 255, size=length, dtype=np.int16))
    a1 = pa.array(np.random.randint(0, 255, size=length, dtype=np.int16))
    batch = pa.record_batch([a0, a1], schema=schema)
    return batch