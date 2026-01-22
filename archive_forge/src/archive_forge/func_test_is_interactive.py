import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test_is_interactive():
    with mock.patch('sys.stdin.isatty', return_value=True):
        assert reauth.is_interactive()