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
def test_middleware_mapping():
    """Test that middleware records methods correctly."""
    server_middleware = RecordingServerMiddlewareFactory()
    client_middleware = RecordingClientMiddlewareFactory()
    with FlightServerBase(middleware={'test': server_middleware}) as server, FlightClient(('localhost', server.port), middleware=[client_middleware]) as client:
        descriptor = flight.FlightDescriptor.for_command(b'')
        with pytest.raises(NotImplementedError):
            list(client.list_flights())
        with pytest.raises(NotImplementedError):
            client.get_flight_info(descriptor)
        with pytest.raises(NotImplementedError):
            client.get_schema(descriptor)
        with pytest.raises(NotImplementedError):
            client.do_get(flight.Ticket(b''))
        with pytest.raises(NotImplementedError):
            writer, _ = client.do_put(descriptor, pa.schema([]))
            writer.close()
        with pytest.raises(NotImplementedError):
            list(client.do_action(flight.Action(b'', b'')))
        with pytest.raises(NotImplementedError):
            list(client.list_actions())
        with pytest.raises(NotImplementedError):
            writer, _ = client.do_exchange(descriptor)
            writer.close()
        expected = [flight.FlightMethod.LIST_FLIGHTS, flight.FlightMethod.GET_FLIGHT_INFO, flight.FlightMethod.GET_SCHEMA, flight.FlightMethod.DO_GET, flight.FlightMethod.DO_PUT, flight.FlightMethod.DO_ACTION, flight.FlightMethod.LIST_ACTIONS, flight.FlightMethod.DO_EXCHANGE]
        assert server_middleware.methods == expected
        assert client_middleware.methods == expected