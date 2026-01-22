import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data({'role': 'admin', 'direction': 'asc'}, {'role': 'admin', 'direction': 'desc'}, {'role': 'user', 'direction': 'asc'}, {'role': 'user', 'direction': 'desc'})
@ddt.unpack
def test_shares_list_with_sorting(self, role, direction):
    self.clients[role].manila('list', params='--sort-key host --sort-dir ' + direction)