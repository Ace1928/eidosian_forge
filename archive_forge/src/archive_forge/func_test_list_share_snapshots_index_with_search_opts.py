from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshots
def test_list_share_snapshots_index_with_search_opts(self):
    search_opts = {'fake_str': 'fake_str_value', 'fake_int': 1}
    cs.share_snapshots.list(detailed=False, search_opts=search_opts)
    cs.assert_called('GET', '/snapshots?fake_int=1&fake_str=fake_str_value')