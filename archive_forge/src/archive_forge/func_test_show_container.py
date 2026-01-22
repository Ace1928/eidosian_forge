from unittest import mock
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.tests.unit.v1 import shell_test_base
from zunclient.v1 import containers_shell
@mock.patch('zunclient.common.cliutils.print_dict')
def test_show_container(self, mock_print_dict):
    fake_container = mock.MagicMock()
    fake_container._info = {}
    fake_container.addresses = {'private': [{'addr': '10.0.0.1'}]}
    containers_shell._show_container(fake_container)
    mock_print_dict.assert_called_once_with({'networks': 'private', 'addresses': '10.0.0.1'})