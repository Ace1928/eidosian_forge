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
def test_none_action_side_effect():
    """Ensure that actions are executed even when we don't consume iterator.

    See https://issues.apache.org/jira/browse/ARROW-14255
    """
    with ActionNoneFlightServer() as server, FlightClient(('localhost', server.port)) as client:
        client.do_action(flight.Action('append', b''))
        r = client.do_action(flight.Action('get_value', b''))
        assert json.loads(next(r).body.to_pybytes()) == [True]