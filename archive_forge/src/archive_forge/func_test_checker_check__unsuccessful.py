from unittest import mock
import requests
from update_checker import UpdateChecker, update_check
@mock.patch('requests.get')
def test_checker_check__unsuccessful(mock_get):
    mock_get.side_effect = requests.exceptions.RequestException
    checker = UpdateChecker(bypass_cache=True)
    assert checker.check(PACKAGE, '1.0.0') is None