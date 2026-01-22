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
@pytest.fixture(params=[pytest.param('file_fixture', id='File Format'), pytest.param('stream_fixture', id='Stream Format')])
def format_fixture(request):
    return request.getfixturevalue(request.param)