import datetime
import os
import sys
import mock
import pytest  # type: ignore
from six.moves import reload_module
from google.auth import _oauth2client
def test_convert_success():
    convert_function = mock.Mock(spec=['__call__'])
    conversion_map_patch = mock.patch.object(_oauth2client, '_CLASS_CONVERSION_MAP', {FakeCredentials: convert_function})
    credentials = FakeCredentials()
    with conversion_map_patch:
        result = _oauth2client.convert(credentials)
    convert_function.assert_called_once_with(credentials)
    assert result == convert_function.return_value