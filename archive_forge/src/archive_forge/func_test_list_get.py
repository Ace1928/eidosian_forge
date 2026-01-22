from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
def test_list_get(self):
    self.manager._list('/human_resources', 'human_resources')
    self.manager.client.get.assert_called_with('/human_resources')