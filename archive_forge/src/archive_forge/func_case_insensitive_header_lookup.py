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
def case_insensitive_header_lookup(headers, lookup_key):
    """Lookup the value of given key in the given headers.
       The key lookup is case-insensitive.
    """
    for key in headers:
        if key.lower() == lookup_key.lower():
            return headers.get(key)