from unittest import mock
import requests
from update_checker import UpdateChecker, update_check
@mock.patch('requests.get')
def test_checker_check__update_to_beta_version_from_beta_version(mock_get):
    mock_response(mock_get.return_value, '4.0.0b5')
    checker = UpdateChecker(bypass_cache=True)
    result = checker.check(PACKAGE, '4.0.0b4')
    assert result.available_version == '4.0.0b5'