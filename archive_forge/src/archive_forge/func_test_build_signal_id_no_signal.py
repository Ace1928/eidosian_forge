from unittest import mock
import swiftclient.client
import testscenarios
import testtools
from testtools import matchers
import time
from heatclient.common import deployment_utils
from heatclient import exc
from heatclient.v1 import software_configs
def test_build_signal_id_no_signal(self):
    hc = mock.MagicMock()
    args = mock.MagicMock()
    args.signal_transport = 'NO_SIGNAL'
    self.assertIsNone(deployment_utils.build_signal_id(hc, args))