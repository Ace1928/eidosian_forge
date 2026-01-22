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
def test_open_stream_with_wrong_options(stream_fixture):
    stream_fixture.write_batches()
    source = stream_fixture.get_source()
    with pytest.raises(TypeError):
        pa.ipc.open_stream(source, options=True)