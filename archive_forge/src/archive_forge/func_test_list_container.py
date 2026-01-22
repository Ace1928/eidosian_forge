from unittest import mock
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.tests.unit.v1 import shell_test_base
from zunclient.v1 import containers_shell
@mock.patch('zunclient.common.cliutils.print_list')
def test_list_container(self, mock_print_list):
    fake_container = mock.MagicMock()
    fake_container._info = {}
    fake_container.addresses = {'private': [{'addr': '10.0.0.1'}]}
    zun_utils.list_containers([fake_container])
    self.assertTrue(mock_print_list.called)
    self.assertEqual(fake_container.addresses, '10.0.0.1')