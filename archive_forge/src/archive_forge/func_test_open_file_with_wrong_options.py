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
def test_open_file_with_wrong_options(file_fixture):
    file_fixture.write_batches()
    source = file_fixture.get_source()
    with pytest.raises(TypeError):
        pa.ipc.open_file(source, options=True)