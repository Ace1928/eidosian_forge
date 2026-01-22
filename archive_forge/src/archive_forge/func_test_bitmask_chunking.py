import numpy as np
import pandas
import pyarrow as pa
import pytest
import modin.pandas as pd
from modin.core.dataframe.pandas.interchange.dataframe_protocol.from_dataframe import (
from modin.pandas.io import from_arrow, from_dataframe
from modin.tests.pandas.utils import df_equals
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import export_frame, get_data_of_all_types, split_df_into_chunks
def test_bitmask_chunking():
    """Test that making a virtual chunk in a middle of a byte of a bitmask doesn't cause problems."""
    at = pa.Table.from_pydict({'col': [True, False, True, True, False] * 5})
    assert at['col'].type.bit_width == 1
    md_df = from_arrow(at)
    exported_df = export_frame(md_df, n_chunks=2)
    df_equals(md_df, exported_df)