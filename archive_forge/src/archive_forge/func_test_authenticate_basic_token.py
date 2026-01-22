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
def test_authenticate_basic_token():
    """Test authenticate_basic_token with bearer token and auth headers."""
    with HeaderAuthFlightServer(auth_handler=no_op_auth_handler, middleware={'auth': HeaderAuthServerMiddlewareFactory()}) as server, FlightClient(('localhost', server.port)) as client:
        token_pair = client.authenticate_basic_token(b'test', b'password')
        assert token_pair[0] == b'authorization'
        assert token_pair[1] == b'Bearer token1234'