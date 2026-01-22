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
def test_partition_keys():
    a, b, c = [ds.field(f) == f for f in 'abc']
    assert ds.get_partition_keys(a) == {'a': 'a'}
    assert ds.get_partition_keys(a) == ds._get_partition_keys(a)
    assert ds.get_partition_keys(a & b & c) == {f: f for f in 'abc'}
    nope = ds.field('d') >= 3
    assert ds.get_partition_keys(nope) == {}
    assert ds.get_partition_keys(a & nope) == {'a': 'a'}
    null = ds.field('a').is_null()
    assert ds.get_partition_keys(null) == {'a': None}