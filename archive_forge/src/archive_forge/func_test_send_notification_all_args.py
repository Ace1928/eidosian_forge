import fixtures
from unittest import mock
from oslo_config import cfg
from stevedore import driver
import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
def test_send_notification_all_args(self):
    t = transport.Transport(_FakeDriver(cfg.CONF))
    t._driver.send_notification = mock.Mock()
    t._send_notification(self._target, 'ctxt', 'message', version=1.0, retry=5)
    t._driver.send_notification.assert_called_once_with(self._target, 'ctxt', 'message', 1.0, retry=5)