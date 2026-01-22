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
def test_message_read_schema(example_messages):
    batches, messages = example_messages
    schema = pa.ipc.read_schema(messages[0])
    assert schema.equals(batches[1].schema)