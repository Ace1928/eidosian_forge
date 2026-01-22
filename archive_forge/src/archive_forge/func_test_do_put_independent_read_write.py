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
def test_do_put_independent_read_write():
    """Ensure that separate threads can read/write on a DoPut."""
    data = [pa.array([-10, -5, 0, 5, 10])]
    table = pa.Table.from_arrays(data, names=['a'])
    with MetadataFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        writer, metadata_reader = client.do_put(flight.FlightDescriptor.for_path(''), table.schema)
        count = [0]

        def _reader_thread():
            while metadata_reader.read() is not None:
                count[0] += 1
        thread = threading.Thread(target=_reader_thread)
        thread.start()
        batches = table.to_batches(max_chunksize=1)
        with writer:
            for idx, batch in enumerate(batches):
                metadata = struct.pack('<i', idx)
                writer.write_with_metadata(batch, metadata)
            writer.done_writing()
            thread.join()
        assert count[0] == len(batches)