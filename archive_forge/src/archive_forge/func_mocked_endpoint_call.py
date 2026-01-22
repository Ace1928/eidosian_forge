import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def mocked_endpoint_call(i, ctxt):
    return mock.call(ctxt, 'testpublisher', 'an_event.start', 'test message%d' % i, {'timestamp': mock.ANY, 'message_id': mock.ANY})