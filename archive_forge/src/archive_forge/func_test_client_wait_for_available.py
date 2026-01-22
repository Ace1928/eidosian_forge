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
def test_client_wait_for_available():
    location = ('localhost', find_free_port())
    server = None

    def serve():
        global server
        time.sleep(0.5)
        server = FlightServerBase(location)
        server.serve()
    with FlightClient(location) as client:
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        started = time.time()
        client.wait_for_available(timeout=5)
        elapsed = time.time() - started
        assert elapsed >= 0.5