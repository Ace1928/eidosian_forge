import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_a_function_callback(self):

    def callback(*args, **kwargs):
        pass
    resource_type = 'thing'
    with mock.patch('keystone.notifications.LOG', self.mock_log):
        notifications.register_event_callback(CREATED_OPERATION, resource_type, callback)
    callback = 'keystone.tests.unit.common.test_notifications.callback'
    expected_log_data = {'callback': callback, 'event': 'identity.%s.created' % resource_type}
    self.verify_log_message([expected_log_data])