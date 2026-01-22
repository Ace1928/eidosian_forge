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
def test_authenticate_basic_token_with_client_middleware():
    """Test authenticate_basic_token with client middleware
       to intercept authorization header returned by the
       HTTP header auth enabled server.
    """
    with HeaderAuthFlightServer(auth_handler=no_op_auth_handler, middleware={'auth': HeaderAuthServerMiddlewareFactory()}) as server:
        client_auth_middleware = ClientHeaderAuthMiddlewareFactory()
        client = FlightClient(('localhost', server.port), middleware=[client_auth_middleware])
        encoded_credentials = base64.b64encode(b'test:password')
        options = flight.FlightCallOptions(headers=[(b'authorization', b'Basic ' + encoded_credentials)])
        result = list(client.do_action(action=flight.Action('test-action', b''), options=options))
        assert result[0].body.to_pybytes() == b'token1234'
        assert client_auth_middleware.call_credential[0] == b'authorization'
        assert client_auth_middleware.call_credential[1] == b'Bearer ' + b'token1234'
        result2 = list(client.do_action(action=flight.Action('test-action', b''), options=options))
        assert result2[0].body.to_pybytes() == b'token1234'
        assert client_auth_middleware.call_credential[0] == b'authorization'
        assert client_auth_middleware.call_credential[1] == b'Bearer ' + b'token1234'
        client.close()