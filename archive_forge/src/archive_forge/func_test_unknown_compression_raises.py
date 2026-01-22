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
def test_unknown_compression_raises():
    with pytest.raises(ValueError):
        Codec.is_available('unknown')
    with pytest.raises(TypeError):
        Codec(None)
    with pytest.raises(ValueError):
        Codec('unknown')