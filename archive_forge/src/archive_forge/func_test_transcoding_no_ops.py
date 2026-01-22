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
@pytest.mark.parametrize('src_encoding, dest_encoding', [('utf-8', 'utf-8'), ('utf-8', 'UTF8')])
def test_transcoding_no_ops(src_encoding, dest_encoding):
    stream = pa.BufferReader(b'abc123')
    assert pa.transcoding_input_stream(stream, src_encoding, dest_encoding) is stream