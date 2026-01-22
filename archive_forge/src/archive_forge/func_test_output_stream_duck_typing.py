import bz2
from contextlib import contextmanager
from io import (BytesIO, StringIO, TextIOWrapper, BufferedIOBase, IOBase)
import itertools
import gc
import gzip
import math
import os
import pathlib
import pytest
import sys
import tempfile
import weakref
import numpy as np
from pyarrow.util import guid
from pyarrow import Codec
import pyarrow as pa
def test_output_stream_duck_typing():

    class DuckWriter:

        def __init__(self):
            self.buf = pa.BufferOutputStream()

        def close(self):
            pass

        @property
        def closed(self):
            return False

        def write(self, data):
            self.buf.write(data)
    duck_writer = DuckWriter()
    stream = pa.output_stream(duck_writer)
    assert stream.write(b'hello')
    assert duck_writer.buf.getvalue().to_pybytes() == b'hello'