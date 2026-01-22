from unittest import mock
from openstack import config
from ironicclient import client as iroclient
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import client as v1
def test_get_client_only_session_passed(self):
    session = mock.Mock()
    session.get_endpoint.return_value = 'http://localhost:35357/v2.0'
    kwargs = {'session': session}
    iroclient.get_client('1', **kwargs)
    session.get_endpoint.assert_called_once_with(service_type='baremetal', interface=None, region_name=None)