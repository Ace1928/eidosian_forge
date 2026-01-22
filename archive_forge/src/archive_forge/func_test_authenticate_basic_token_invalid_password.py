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
def test_authenticate_basic_token_invalid_password():
    """Test authenticate_basic_token with an invalid password."""
    with HeaderAuthFlightServer(auth_handler=no_op_auth_handler, middleware={'auth': HeaderAuthServerMiddlewareFactory()}) as server, FlightClient(('localhost', server.port)) as client:
        with pytest.raises(flight.FlightUnauthenticatedError):
            client.authenticate_basic_token(b'test', b'badpassword')