import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_to_bytes_with_unicode():
    value = u'string-val'
    encoded_value = b'string-val'
    assert _helpers.to_bytes(value) == encoded_value