import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
def test_open_union_dataset_with_additional_kwargs(multisourcefs):
    child = ds.dataset('/plain', filesystem=multisourcefs, format='parquet')
    with pytest.raises(ValueError, match='cannot pass any additional'):
        ds.dataset([child], format='parquet')