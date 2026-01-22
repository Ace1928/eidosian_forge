import fixtures
from unittest import mock
from oslo_config import cfg
from stevedore import driver
import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
def test_send_all_args(self):
    t = transport.Transport(_FakeDriver(cfg.CONF))
    t._driver.send = mock.Mock()
    t._send(self._target, 'ctxt', 'message', wait_for_reply='wait_for_reply', timeout='timeout', call_monitor_timeout='cm_timeout', retry='retry')
    t._driver.send.assert_called_once_with(self._target, 'ctxt', 'message', wait_for_reply='wait_for_reply', timeout='timeout', call_monitor_timeout='cm_timeout', retry='retry', transport_options=None)