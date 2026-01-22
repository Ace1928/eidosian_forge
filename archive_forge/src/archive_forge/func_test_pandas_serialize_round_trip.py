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
def test_pandas_serialize_round_trip():
    index = pd.Index([1, 2, 3], name='my_index')
    columns = ['foo', 'bar']
    df = pd.DataFrame({'foo': [1.5, 1.6, 1.7], 'bar': list('abc')}, index=index, columns=columns)
    _check_serialize_pandas_round_trip(df)