import errno
import os
import subprocess
from unittest import mock
from heat.cloudinit import loguserdata
from heat.tests import common
@mock.patch('sys.exc_info')
@mock.patch('subprocess.Popen')
def test_call_exception(self, mock_popen, mock_exc_info):
    mock_popen.side_effect = Exception()
    no_exec = mock.MagicMock(errno='irrelevant')
    mock_exc_info.return_value = (None, no_exec, None)
    return_code = loguserdata.call(['foo', 'bar'])
    self.assertEqual(os.EX_SOFTWARE, return_code)