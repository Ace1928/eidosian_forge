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
@pytest.mark.slow
def test_write_error_propagation():
    """
    Ensure that exceptions during writing preserve error context.

    See https://issues.apache.org/jira/browse/ARROW-16592.
    """
    expected_message = 'foo'
    expected_info = b'bar'
    exc = flight.FlightCancelledError(expected_message, extra_info=expected_info)
    descriptor = flight.FlightDescriptor.for_command(b'')
    schema = pa.schema([('int64', pa.int64())])

    class FailServer(flight.FlightServerBase):

        def do_put(self, context, descriptor, reader, writer):
            raise exc

        def do_exchange(self, context, descriptor, reader, writer):
            raise exc
    with FailServer() as server, FlightClient(('localhost', server.port)) as client:
        writer, reader = client.do_put(descriptor, schema)

        def _reader():
            try:
                while True:
                    reader.read()
            except flight.FlightError:
                return
        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        with pytest.raises(flight.FlightCancelledError) as exc_info:
            while True:
                writer.write_batch(pa.record_batch([[1]], schema=schema))
        assert exc_info.value.extra_info == expected_info
        with pytest.raises(flight.FlightCancelledError) as exc_info:
            writer.close()
        assert exc_info.value.extra_info == expected_info
        thread.join()
        writer, reader = client.do_exchange(descriptor)

        def _reader():
            try:
                while True:
                    reader.read_chunk()
            except flight.FlightError:
                return
        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        with pytest.raises(flight.FlightCancelledError) as exc_info:
            while True:
                writer.write_metadata(b' ')
        assert exc_info.value.extra_info == expected_info
        with pytest.raises(flight.FlightCancelledError) as exc_info:
            writer.close()
        assert exc_info.value.extra_info == expected_info
        thread.join()