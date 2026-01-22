import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import pyarrow as pa
def pandas_compatible_list_types(item_strategy=pandas_compatible_primitive_types):
    return st.builds(pa.list_, item_strategy) | st.builds(pa.large_list, item_strategy)