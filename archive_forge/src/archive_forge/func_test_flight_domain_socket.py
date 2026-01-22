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
@pytest.mark.skipif(os.name == 'nt', reason="Unix sockets can't be tested on Windows")
def test_flight_domain_socket():
    """Try a simple do_get call over a Unix domain socket."""
    with tempfile.NamedTemporaryFile() as sock:
        sock.close()
        location = flight.Location.for_grpc_unix(sock.name)
        with ConstantFlightServer(location=location), FlightClient(location) as client:
            reader = client.do_get(flight.Ticket(b'ints'))
            table = simple_ints_table()
            assert reader.schema.equals(table.schema)
            data = reader.read_all()
            assert data.equals(table)
            reader = client.do_get(flight.Ticket(b'dicts'))
            table = simple_dicts_table()
            assert reader.schema.equals(table.schema)
            data = reader.read_all()
            assert data.equals(table)