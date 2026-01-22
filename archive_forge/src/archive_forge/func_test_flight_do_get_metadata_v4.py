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
def test_flight_do_get_metadata_v4():
    """Try a simple do_get call with V4 metadata version."""
    table = pa.Table.from_arrays([pa.array([-10, -5, 0, 5, 10])], names=['a'])
    options = pa.ipc.IpcWriteOptions(metadata_version=pa.ipc.MetadataVersion.V4)
    with MetadataFlightServer(options=options) as server, FlightClient(('localhost', server.port)) as client:
        reader = client.do_get(flight.Ticket(b''))
        data = reader.read_all()
        assert data.equals(table)