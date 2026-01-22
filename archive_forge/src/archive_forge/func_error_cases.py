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
@staticmethod
def error_cases():
    return {'internal': flight.FlightInternalError, 'timedout': flight.FlightTimedOutError, 'cancel': flight.FlightCancelledError, 'unauthenticated': flight.FlightUnauthenticatedError, 'unauthorized': flight.FlightUnauthorizedError, 'notimplemented': NotImplementedError, 'invalid': pa.ArrowInvalid, 'key': KeyError}