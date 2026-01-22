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
@pytest.mark.requires_testing_data
def test_tls_fails():
    """Make sure clients cannot connect when cert verification fails."""
    certs = example_tls_certs()
    with ConstantFlightServer(tls_certificates=certs['certificates']) as s, FlightClient('grpc+tls://localhost:' + str(s.port)) as client:
        with pytest.raises(flight.FlightUnavailableError):
            client.do_get(flight.Ticket(b'ints')).read_all()