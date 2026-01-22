from unittest.mock import patch
from pyct.report import report
@patch('importlib.import_module')
@patch('builtins.print')
def test_report_gives_package_version(mock_print, mock_import_module):
    module = TestModule()
    mock_import_module.return_value = module
    report('param')
    mock_print.assert_called_with('param=1.9.3                    # /mock/opt/anaconda3/envs/pyct/lib/python3.7/site-packages/param')