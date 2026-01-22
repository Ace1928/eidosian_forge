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
def test_flight_invalid_generator_stream():
    """Try streaming data with mismatched schemas."""
    with InvalidStreamFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        with pytest.raises(pa.ArrowException):
            client.do_get(flight.Ticket(b'')).read_all()