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
def test_flight_generator_stream():
    """Try downloading a flight of RecordBatches in a GeneratorStream."""
    data = pa.Table.from_arrays([pa.array(range(0, 10 * 1024))], names=['a'])
    with EchoStreamFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        writer, _ = client.do_put(flight.FlightDescriptor.for_path('test'), data.schema)
        writer.write_table(data)
        writer.close()
        result = client.do_get(flight.Ticket(b'')).read_all()
        assert result.equals(data)