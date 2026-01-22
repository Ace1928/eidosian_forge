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
def test_token_auth_invalid():
    """Test an auth mechanism that uses a handshake."""
    with EchoStreamFlightServer(auth_handler=token_auth_handler) as server, FlightClient(('localhost', server.port)) as client:
        with pytest.raises(flight.FlightUnauthenticatedError):
            client.authenticate(TokenClientAuthHandler('test', 'wrong'))