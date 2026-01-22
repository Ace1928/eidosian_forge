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
def assert_yields_projected(fragment, row_slice, columns=None, filter=None):
    actual = fragment.to_table(schema=table.schema, columns=columns, filter=filter)
    column_names = columns if columns else table.column_names
    assert actual.column_names == column_names
    expected = table.slice(*row_slice).select(column_names)
    assert actual.equals(expected)