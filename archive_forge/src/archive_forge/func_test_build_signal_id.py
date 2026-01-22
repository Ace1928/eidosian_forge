from unittest import mock
import swiftclient.client
import testscenarios
import testtools
from testtools import matchers
import time
from heatclient.common import deployment_utils
from heatclient import exc
from heatclient.v1 import software_configs
@mock.patch.object(deployment_utils, 'create_temp_url')
@mock.patch.object(deployment_utils, 'create_swift_client')
def test_build_signal_id(self, csc, ctu):
    hc = mock.MagicMock()
    args = mock.MagicMock()
    args.name = 'foo'
    args.timeout = 60
    args.os_no_client_auth = False
    args.signal_transport = 'TEMP_URL_SIGNAL'
    csc.return_value = mock.MagicMock()
    temp_url = 'http://fake-host.com:8080/v1/AUTH_demo/foo/a81a74d5-c395-4269-9670-ddd0824fd696?temp_url_sig=6a68371d602c7a14aaaa9e3b3a63b8b85bd9a503&temp_url_expires=1425270977'
    ctu.return_value = temp_url
    self.assertEqual(temp_url, deployment_utils.build_signal_id(hc, args))
    self.assertEqual(mock.call(hc.http_client.auth, hc.http_client.session, args), csc.call_args)
    self.assertEqual(mock.call(csc.return_value, 'foo', 60), ctu.call_args)