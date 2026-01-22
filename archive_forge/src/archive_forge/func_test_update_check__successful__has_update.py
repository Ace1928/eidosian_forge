from unittest import mock
import requests
from update_checker import UpdateChecker, update_check
@mock.patch('requests.get')
def test_update_check__successful__has_update(mock_get, capsys):
    mock_response(mock_get.return_value)
    update_check(PACKAGE, '0.0.1', bypass_cache=True)
    assert 'Version 0.0.1 of praw is outdated. Version 5.0.0 is available.\n' == capsys.readouterr().err