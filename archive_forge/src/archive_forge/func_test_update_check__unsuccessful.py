from unittest import mock
import requests
from update_checker import UpdateChecker, update_check
@mock.patch('requests.get')
def test_update_check__unsuccessful(mock_get, capsys):
    mock_get.side_effect = requests.exceptions.RequestException
    update_check(PACKAGE, '0.0.1', bypass_cache=True)
    assert '' == capsys.readouterr().err