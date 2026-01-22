import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_to_bytes_with_bytes():
    value = b'bytes-val'
    assert _helpers.to_bytes(value) == value