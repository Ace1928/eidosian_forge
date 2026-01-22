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
def test_flight_do_get_ints():
    """Try a simple do_get call."""
    table = simple_ints_table()
    with ConstantFlightServer() as server, flight.connect(('localhost', server.port)) as client:
        data = client.do_get(flight.Ticket(b'ints')).read_all()
        assert data.equals(table)
    options = pa.ipc.IpcWriteOptions(metadata_version=pa.ipc.MetadataVersion.V4)
    with ConstantFlightServer(options=options) as server, flight.connect(('localhost', server.port)) as client:
        data = client.do_get(flight.Ticket(b'ints')).read_all()
        assert data.equals(table)
        data = client.do_get(flight.Ticket(b'ints')).to_reader().read_all()
        assert data.equals(table)
    with pytest.raises(flight.FlightServerError, match="expected IpcWriteOptions, got <class 'int'>"):
        with ConstantFlightServer(options=42) as server, flight.connect(('localhost', server.port)) as client:
            data = client.do_get(flight.Ticket(b'ints')).read_all()