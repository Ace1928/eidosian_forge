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
def exchange_echo(self, context, reader, writer):
    """Run a simple echo server."""
    started = False
    for chunk in reader:
        if not started and chunk.data:
            writer.begin(chunk.data.schema, options=self.options)
            started = True
        if chunk.app_metadata and chunk.data:
            writer.write_with_metadata(chunk.data, chunk.app_metadata)
        elif chunk.app_metadata:
            writer.write_metadata(chunk.app_metadata)
        elif chunk.data:
            writer.write_batch(chunk.data)
        else:
            assert False, 'Should not happen'