from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
def test_list_shares_index_with_search_opts(self):
    search_opts = {'fake_str': 'fake_str_value', 'fake_int': 1, 'name~': 'fake_name', 'description~': 'fake_description'}
    cs.shares.list(detailed=False, search_opts=search_opts)
    cs.assert_called('GET', '/shares?description~=fake_description&fake_int=1&fake_str=fake_str_value&is_public=True&name~=fake_name')