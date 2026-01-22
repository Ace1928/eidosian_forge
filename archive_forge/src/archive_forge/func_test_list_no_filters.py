from unittest import mock
import ddt
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import security_services
def test_list_no_filters(self):
    with mock.patch.object(self.manager, '_list', mock.Mock(return_value=None)):
        self.manager.list()
        self.manager._list.assert_called_once_with(security_services.RESOURCES_PATH + '/detail', security_services.RESOURCES_NAME)