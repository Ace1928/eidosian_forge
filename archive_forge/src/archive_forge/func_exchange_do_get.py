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
def exchange_do_get(self, context, reader, writer):
    """Emulate DoGet with DoExchange."""
    data = pa.Table.from_arrays([pa.array(range(0, 10 * 1024))], names=['a'])
    writer.begin(data.schema)
    writer.write_table(data)