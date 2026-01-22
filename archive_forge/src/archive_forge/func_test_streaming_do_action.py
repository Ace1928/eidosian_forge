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
def test_streaming_do_action():
    with ConvenienceServer() as server, FlightClient(('localhost', server.port)) as client:
        results = client.do_action(flight.Action('forever', b''))
        assert next(results).body == b'foo'
        del results