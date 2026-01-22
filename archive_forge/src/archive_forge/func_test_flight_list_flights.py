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
def test_flight_list_flights():
    """Try a simple list_flights call."""
    with ConstantFlightServer() as server, flight.connect(('localhost', server.port)) as client:
        assert list(client.list_flights()) == []
        flights = client.list_flights(ConstantFlightServer.CRITERIA)
        assert len(list(flights)) == 1