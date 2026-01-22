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
def test_token_auth():
    """Test an auth mechanism that uses a handshake."""
    with EchoStreamFlightServer(auth_handler=token_auth_handler) as server, FlightClient(('localhost', server.port)) as client:
        action = flight.Action('who-am-i', b'')
        client.authenticate(TokenClientAuthHandler('test', 'p4ssw0rd'))
        identity = next(client.do_action(action))
        assert identity.body.to_pybytes() == b'test'