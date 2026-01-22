from unittest import mock
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.tests.unit.v1 import shell_test_base
from zunclient.v1 import containers_shell
@mock.patch('zunclient.v1.containers_shell._show_container')
@mock.patch('zunclient.v1.containers.ContainerManager.create')
def test_zun_container_create_success(self, mock_create, mock_show_container):
    mock_create.return_value = 'container'
    self._test_arg_success('create x')
    mock_show_container.assert_called_once_with('container')