import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def wrong_signature():
    return pa.scalar(1, type=pa.int64())