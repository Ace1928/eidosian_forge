import os
import random
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _test_dataframe
from pyarrow.tests.parquet.test_dataset import (
from pyarrow.util import guid
def hdfs_test_client():
    host = os.environ.get('ARROW_HDFS_TEST_HOST', 'default')
    user = os.environ.get('ARROW_HDFS_TEST_USER', None)
    try:
        port = int(os.environ.get('ARROW_HDFS_TEST_PORT', 0))
    except ValueError:
        raise ValueError('Env variable ARROW_HDFS_TEST_PORT was not an integer')
    with pytest.warns(FutureWarning):
        return pa.hdfs.connect(host, port, user)