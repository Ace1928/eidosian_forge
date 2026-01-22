from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from unittest import mock
from gslib import command
from gslib.tests import testcase
from gslib.utils import constants
@mock.patch.object(command, 'CreateOrGetGsutilLogger', autospec=True)
def test_quiet_mode_gets_set(self, mock_logger):
    mock_logger.return_value.isEnabledFor.return_value = False
    self._fake_command = FakeGsutilCommand(command_runner=mock.ANY, args=['-z', 'opt1', '-r', 'arg1', 'arg2'], headers={}, debug=mock.ANY, trace_token=mock.ANY, parallel_operations=mock.ANY, bucket_storage_uri_class=mock.ANY, gsutil_api_class_map_factory=mock.MagicMock())
    self.assertTrue(self._fake_command.quiet_mode)
    mock_logger.assert_called_once_with('fake_gsutil')