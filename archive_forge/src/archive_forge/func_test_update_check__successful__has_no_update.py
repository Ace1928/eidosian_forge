from unittest import mock
import requests
from update_checker import UpdateChecker, update_check
@mock.patch('requests.get')
def test_update_check__successful__has_no_update(mock_get, capsys):
    mock_response(mock_get.return_value, '0.0.2')
    update_check(PACKAGE, '0.0.2', bypass_cache=True)
    assert '' == capsys.readouterr().err