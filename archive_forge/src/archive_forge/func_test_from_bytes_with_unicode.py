import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_from_bytes_with_unicode():
    value = u'bytes-val'
    assert _helpers.from_bytes(value) == value