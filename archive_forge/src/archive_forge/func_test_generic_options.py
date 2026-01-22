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
@pytest.mark.requires_testing_data
def test_generic_options():
    """Test setting generic client options."""
    certs = example_tls_certs()
    with ConstantFlightServer(tls_certificates=certs['certificates']) as s:
        options = [('grpc.ssl_target_name_override', 'fakehostname')]
        client = flight.connect(('localhost', s.port), tls_root_certs=certs['root_cert'], generic_options=options)
        with pytest.raises(flight.FlightUnavailableError):
            client.do_get(flight.Ticket(b'ints'))
        client.close()
        options = [('grpc.max_receive_message_length', 32)]
        client = flight.connect(('localhost', s.port), tls_root_certs=certs['root_cert'], generic_options=options)
        with pytest.raises((pa.ArrowInvalid, flight.FlightCancelledError)):
            client.do_get(flight.Ticket(b'ints'))
        client.close()