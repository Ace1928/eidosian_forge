import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_from_bytes_with_bytes():
    value = b'string-val'
    decoded_value = u'string-val'
    assert _helpers.from_bytes(value) == decoded_value