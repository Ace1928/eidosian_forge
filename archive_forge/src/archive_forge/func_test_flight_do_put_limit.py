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
def test_flight_do_put_limit():
    """Try a simple do_put call with a size limit."""
    large_batch = pa.RecordBatch.from_arrays([pa.array(np.ones(768, dtype=np.int64()))], names=['a'])
    with EchoFlightServer() as server, FlightClient(('localhost', server.port), write_size_limit_bytes=4096) as client:
        writer, metadata_reader = client.do_put(flight.FlightDescriptor.for_path(''), large_batch.schema)
        with writer:
            with pytest.raises(flight.FlightWriteSizeExceededError, match='exceeded soft limit') as excinfo:
                writer.write_batch(large_batch)
            assert excinfo.value.limit == 4096
            smaller_batches = [large_batch.slice(0, 384), large_batch.slice(384)]
            for batch in smaller_batches:
                writer.write_batch(batch)
        expected = pa.Table.from_batches([large_batch])
        actual = client.do_get(flight.Ticket(b'')).read_all()
        assert expected == actual