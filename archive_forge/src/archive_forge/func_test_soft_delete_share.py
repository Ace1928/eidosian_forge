from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
def test_soft_delete_share(self):
    share = cs.shares.get('1234')
    cs.shares.soft_delete(share)
    body = {'soft_delete': None}
    cs.assert_called('POST', '/shares/1234/action', body=body)