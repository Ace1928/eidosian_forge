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
def test_flight_get_info():
    """Make sure FlightEndpoint accepts string and object URIs."""
    with GetInfoFlightServer() as server:
        client = FlightClient(('localhost', server.port))
        info = client.get_flight_info(flight.FlightDescriptor.for_command(b''))
        assert info.total_records == -1
        assert info.total_bytes == -1
        assert info.schema == pa.schema([('a', pa.int32())])
        assert len(info.endpoints) == 2
        assert len(info.endpoints[0].locations) == 1
        assert info.endpoints[0].locations[0] == flight.Location('grpc://test')
        assert info.endpoints[1].locations[0] == flight.Location.for_grpc_tcp('localhost', 5005)