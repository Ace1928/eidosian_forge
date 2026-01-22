from unittest import mock
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import scheduler_stats
@mock.patch.object(scheduler_stats.PoolManager, '_list', mock.Mock())
def test_list_detail(self):
    self.manager.list()
    self.manager._list.assert_called_once_with(scheduler_stats.RESOURCES_PATH + '/detail', scheduler_stats.RESOURCES_NAME)