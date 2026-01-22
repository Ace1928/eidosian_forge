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
def test_location_unknown_scheme():
    """Test creating locations for unknown schemes."""
    assert flight.Location('s3://foo').uri == b's3://foo'
    assert flight.Location('https://example.com/bar.parquet').uri == b'https://example.com/bar.parquet'