from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import client
from manilaclient import exceptions
from manilaclient.tests.unit import utils
import manilaclient.v1.client
import manilaclient.v2.client
def test_init_client_with_api_version_instance(self):
    version = manilaclient.API_MAX_VERSION
    with mock.patch.object(manilaclient.v2.client, 'Client'):
        manilaclient.client.Client(version, 'foo', bar='quuz')
        manilaclient.v2.client.Client.assert_called_once_with('foo', api_version=version, bar='quuz')