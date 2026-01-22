from unittest.mock import patch
from pyct.report import report
@patch('builtins.print')
@patch('subprocess.check_output')
def test_report_gives_conda_version(mock_check_output, mock_print):
    mock_check_output.side_effect = [b'/mock/opt/anaconda3/condabin/conda\n', b'conda 4.8.3\n']
    report('conda')
    mock_print.assert_called_with('conda=4.8.3                    # /mock/opt/anaconda3/condabin/conda')