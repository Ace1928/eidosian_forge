from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
def test_list_shares_sort_by_desc_and_size_key(self):
    cs.shares.list(detailed=False, sort_key='size', sort_dir='desc')
    cs.assert_called('GET', '/shares?is_public=True&sort_dir=desc&sort_key=size')