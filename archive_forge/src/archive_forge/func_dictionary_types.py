import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import pyarrow as pa
def dictionary_types(key_strategy=None, value_strategy=None):
    if key_strategy is None:
        key_strategy = signed_integer_types
    if value_strategy is None:
        value_strategy = st.one_of(bool_type, integer_types, st.sampled_from([pa.float32(), pa.float64()]), binary_type, string_type, fixed_size_binary_type)
    return st.builds(pa.dictionary, key_strategy, value_strategy)