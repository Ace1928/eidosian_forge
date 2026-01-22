import datetime
import os
import sys
import mock
import pytest  # type: ignore
from six.moves import reload_module
from google.auth import _oauth2client
def test_import_without_oauth2client(monkeypatch, reset__oauth2client_module):
    monkeypatch.setitem(sys.modules, 'oauth2client', None)
    with pytest.raises(ImportError) as excinfo:
        reload_module(_oauth2client)
    assert excinfo.match('oauth2client')