import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_to_bytes_with_nonstring_type():
    with pytest.raises(ValueError):
        _helpers.to_bytes(object())