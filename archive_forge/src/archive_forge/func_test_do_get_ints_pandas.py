import ast
import base64
import itertools
import os
import pathlib
import signal
import struct
import tempfile
import threading
import time
import traceback
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.lib import IpcReadOptions, tobytes
from pyarrow.util import find_free_port
from pyarrow.tests import util
@pytest.mark.pandas
def test_do_get_ints_pandas():
    """Try a simple do_get call."""
    table = simple_ints_table()
    with ConstantFlightServer() as server, flight.connect(('localhost', server.port)) as client:
        data = client.do_get(flight.Ticket(b'ints')).read_pandas()
        assert list(data['some_ints']) == table.column(0).to_pylist()