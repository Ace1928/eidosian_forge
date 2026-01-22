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
def example_tls_certs():
    """Get the paths to test TLS certificates."""
    return {'root_cert': read_flight_resource('root-ca.pem'), 'certificates': [flight.CertKeyPair(cert=read_flight_resource('cert0.pem'), key=read_flight_resource('cert0.key')), flight.CertKeyPair(cert=read_flight_resource('cert1.pem'), key=read_flight_resource('cert1.key'))]}