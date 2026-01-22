import datetime
import os
import sys
import mock
import pytest  # type: ignore
from six.moves import reload_module
from google.auth import _oauth2client
def test_convert_not_found():
    with pytest.raises(ValueError) as excinfo:
        _oauth2client.convert('a string is not a real credentials class')
    assert excinfo.match('Unable to convert')