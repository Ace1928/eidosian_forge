from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_snapshots as snapshots
@ddt.data({'sort_key': 'name', 'sort_dir': 'invalid'}, {'sort_key': 'invalid', 'sort_dir': 'asc'})
@ddt.unpack
def test_list_with_invalid_sorting(self, sort_key, sort_dir):
    self.assertRaises(ValueError, self.manager.list, sort_dir=sort_dir, sort_key=sort_key)