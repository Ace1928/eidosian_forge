import argparse
from unittest import mock
import testtools
from ironicclient.osc import plugin
from ironicclient.tests.unit.osc import fakes
from ironicclient.v1 import client
@mock.patch.object(plugin, 'OS_BAREMETAL_API_LATEST', new=False)
@mock.patch.object(client, 'Client', autospec=True)
def test_make_client_v1(self, mock_client):
    instance = fakes.FakeClientManager()
    instance.get_endpoint_for_service_type = mock.Mock(return_value='endpoint')
    instance._api_version = {'baremetal': '1'}
    plugin.make_client(instance)
    mock_client.assert_called_once_with(os_ironic_api_version=plugin.LATEST_VERSION, allow_api_version_downgrade=True, session=instance.session, region_name=instance._region_name, endpoint_override='endpoint')
    instance.get_endpoint_for_service_type.assert_called_once_with('baremetal', region_name=instance._region_name, interface=instance.interface)