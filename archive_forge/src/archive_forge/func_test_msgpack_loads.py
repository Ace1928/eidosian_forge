import pytest
from pathlib import Path
import datetime
from mock import patch
import numpy
from .._msgpack_api import read_msgpack, write_msgpack
from .._msgpack_api import msgpack_loads, msgpack_dumps
from .._msgpack_api import msgpack_encoders, msgpack_decoders
from .util import make_tempdir
def test_msgpack_loads():
    msg = b'\x82\xa5hello\xa5world\xa4test{'
    data = msgpack_loads(msg)
    assert len(data) == 2
    assert data['hello'] == 'world'
    assert data['test'] == 123