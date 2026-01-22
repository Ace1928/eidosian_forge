import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def wrong_output_type(ctx):
    return 42