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
def test_image_get_data_notification(self):
    self.image_proxy.size = 10
    data = ''.join(self.image_proxy.get_data())
    self.assertEqual('0123456789', data)
    output_logs = self.notifier.get_logs()
    self.assertEqual(1, len(output_logs))
    output_log = output_logs[0]
    self.assertEqual('INFO', output_log['notification_type'])
    self.assertEqual('image.send', output_log['event_type'])
    self.assertEqual(self.image.image_id, output_log['payload']['image_id'])
    self.assertEqual(TENANT2, output_log['payload']['receiver_tenant_id'])
    self.assertEqual(USER1, output_log['payload']['receiver_user_id'])
    self.assertEqual(10, output_log['payload']['bytes_sent'])
    self.assertEqual(TENANT1, output_log['payload']['owner_id'])