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
@pytest.mark.large_memory
@pytest.mark.slow
def test_large_metadata_client():
    descriptor = flight.FlightDescriptor.for_command(b'')
    metadata = b' ' * (2 ** 31 + 1)
    with EchoFlightServer() as server, flight.connect(('localhost', server.port)) as client:
        with pytest.raises(pa.ArrowCapacityError, match='app_metadata size overflow'):
            writer, _ = client.do_put(descriptor, pa.schema([]))
            with writer:
                writer.write_metadata(metadata)
                writer.close()
        with pytest.raises(pa.ArrowCapacityError, match='app_metadata size overflow'):
            writer, reader = client.do_exchange(descriptor)
            with writer:
                writer.write_metadata(metadata)
    del metadata
    with LargeMetadataFlightServer() as server, flight.connect(('localhost', server.port)) as client:
        with pytest.raises(flight.FlightServerError, match='app_metadata size overflow'):
            reader = client.do_get(flight.Ticket(b''))
            reader.read_all()
        with pytest.raises(pa.ArrowException, match='app_metadata size overflow'):
            writer, reader = client.do_exchange(descriptor)
            with writer:
                reader.read_all()