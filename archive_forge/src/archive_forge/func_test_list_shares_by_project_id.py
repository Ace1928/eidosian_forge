import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
def test_list_shares_by_project_id(self):
    project_id = self.user_client.get_project_id(self.user_client.tenant_name)
    self._list_shares({'project_id': project_id})