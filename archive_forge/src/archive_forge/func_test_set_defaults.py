import datetime
from unittest import mock
import glance_store
from oslo_config import cfg
import oslo_messaging
import webob
import glance.async_
from glance.common import exception
from glance.common import timeutils
import glance.context
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
from glance.tests import utils
@mock.patch.object(oslo_messaging, 'set_transport_defaults')
def test_set_defaults(self, mock_set_trans_defaults):
    notifier.set_defaults(control_exchange='foo')
    mock_set_trans_defaults.assert_called_with('foo')
    notifier.set_defaults()
    mock_set_trans_defaults.assert_called_with('glance')