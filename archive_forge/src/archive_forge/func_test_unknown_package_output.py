from unittest.mock import patch
from pyct.report import report
@patch('builtins.print')
def test_unknown_package_output(mock_print):
    report('fake_package')
    mock_print.assert_called_with('fake_package=unknown           # not installed in this environment')