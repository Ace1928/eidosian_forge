import pytest
import pyarrow as pa
from pyarrow import Codec
from pyarrow import fs
import numpy as np
def unary_function(ctx, x):
    return pc.call_function('add', [x, 1], memory_pool=ctx.memory_pool)