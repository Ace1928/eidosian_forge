from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.v2 import services
def test_list_services_with_one_search_opt(self):
    manager = self._get_manager('2.7')
    host = 'fake_host'
    query_string = '?host=%s' % host
    with mock.patch.object(manager, '_list', mock.Mock(return_value=None)):
        manager.list({'host': host})
        manager._list.assert_called_once_with(services.RESOURCE_PATH + query_string, services.RESOURCE_NAME)