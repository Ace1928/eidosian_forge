import ast
import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data('host', 'status', 'project_id', 'share_network', 'host,status,project_id,share_network')
def test_share_server_list_with_specified_columns(self, columns):
    self.client.list_share_servers(columns=columns)