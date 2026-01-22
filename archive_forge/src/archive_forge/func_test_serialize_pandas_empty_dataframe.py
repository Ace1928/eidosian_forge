from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
@pytest.mark.pandas
def test_serialize_pandas_empty_dataframe():
    df = pd.DataFrame()
    _check_serialize_pandas_round_trip(df)