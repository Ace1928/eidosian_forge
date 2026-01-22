from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import client
from manilaclient import exceptions
from manilaclient.tests.unit import utils
import manilaclient.v1.client
import manilaclient.v2.client
@ddt.data('1', '1.0')
def test_init_client_with_string_v1_version(self, version):
    with mock.patch.object(manilaclient.v1.client, 'Client'):
        with mock.patch.object(api_versions, 'APIVersion'):
            api_instance = api_versions.APIVersion.return_value
            api_instance.get_major_version.return_value = '1'
            manilaclient.client.Client(version, 'foo', bar='quuz')
            manilaclient.v1.client.Client.assert_called_once_with('foo', api_version=api_instance, bar='quuz')
            api_versions.APIVersion.assert_called_once_with('1.0')