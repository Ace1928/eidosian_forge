from oslo_utils import timeutils
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher as notify_dispatcher
from oslo_messaging.tests import utils as test_utils
from unittest import mock
@mock.patch('oslo_messaging.notify.dispatcher.LOG')
def test_dispatcher_unknown_prio(self, mylog):
    msg = notification_msg.copy()
    msg['priority'] = 'what???'
    dispatcher = notify_dispatcher.NotificationDispatcher([mock.Mock()], None)
    res = dispatcher.dispatch(mock.Mock(ctxt={}, message=msg))
    self.assertIsNone(res)
    mylog.warning.assert_called_once_with('Unknown priority "%s"', 'what???')