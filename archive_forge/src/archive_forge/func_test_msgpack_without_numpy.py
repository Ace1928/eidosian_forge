import pytest
from pathlib import Path
import datetime
from mock import patch
import numpy
from .._msgpack_api import read_msgpack, write_msgpack
from .._msgpack_api import msgpack_loads, msgpack_dumps
from .._msgpack_api import msgpack_encoders, msgpack_decoders
from .util import make_tempdir
@patch('srsly.msgpack._msgpack_numpy.np', None)
@patch('srsly.msgpack._msgpack_numpy.has_numpy', False)
def test_msgpack_without_numpy():
    """Test that msgpack works without numpy and raises correct errors (e.g.
    when serializing datetime objects, the error should be msgpack's TypeError,
    not a "'np' is not defined error")."""
    with pytest.raises(TypeError):
        msgpack_loads(msgpack_dumps(datetime.datetime.now()))