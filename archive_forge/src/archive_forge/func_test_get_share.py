from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
@ddt.data(type('ShareUUID', (object,), {'uuid': '1234'}), type('ShareID', (object,), {'id': '1234'}), '1234')
def test_get_share(self, share):
    share = cs.shares.get(share)
    cs.assert_called('GET', '/shares/1234')