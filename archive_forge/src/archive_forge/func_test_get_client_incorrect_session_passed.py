from unittest import mock
from openstack import config
from ironicclient import client as iroclient
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import client as v1
def test_get_client_incorrect_session_passed(self):
    session = mock.Mock()
    session.get_endpoint.side_effect = Exception('boo')
    kwargs = {'session': session}
    self.assertRaises(exc.AmbiguousAuthSystem, iroclient.get_client, '1', **kwargs)