from unittest import mock
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.tests.unit.v1 import shell_test_base
from zunclient.v1 import containers_shell
@mock.patch('zunclient.v1.containers.ContainerManager.list')
def test_zun_container_command_failure(self, mock_list):
    self._test_arg_failure('list --wrong', self._unrecognized_arg_error)
    self.assertFalse(mock_list.called)