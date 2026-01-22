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
def test_file_simple_roundtrip(file_fixture):
    file_fixture._check_roundtrip(as_table=False)