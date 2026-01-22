import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def pct_rank(ctx, x):
    return pa.array(x.to_pandas().copy().rank(pct=True))