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
def test_image_set_data_upload_and_activate_notification(self):
    image = ImageStub(image_id=UUID1, name='image-1', status='queued', created_at=DATETIME, updated_at=DATETIME, owner=TENANT1, visibility='public')
    context = glance.context.RequestContext(tenant=TENANT2, user=USER1)
    fake_notifier = unit_test_utils.FakeNotifier()
    image_proxy = glance.notifier.ImageProxy(image, context, fake_notifier)

    def data_iterator():
        fake_notifier.log = []
        yield 'abcde'
        yield 'fghij'
        image_proxy.extra_properties['os_glance_importing_to_stores'] = 'store2'
    image_proxy.extra_properties['os_glance_importing_to_stores'] = 'store1,store2'
    image_proxy.extra_properties['os_glance_failed_import'] = ''
    image_proxy.set_data(data_iterator(), 10)
    output_logs = fake_notifier.get_logs()
    self.assertEqual(2, len(output_logs))
    output_log = output_logs[0]
    self.assertEqual('INFO', output_log['notification_type'])
    self.assertEqual('image.upload', output_log['event_type'])
    self.assertEqual(self.image.image_id, output_log['payload']['id'])
    self.assertEqual(['store2'], output_log['payload']['os_glance_importing_to_stores'])
    self.assertEqual([], output_log['payload']['os_glance_failed_import'])
    output_log = output_logs[1]
    self.assertEqual('INFO', output_log['notification_type'])
    self.assertEqual('image.activate', output_log['event_type'])
    self.assertEqual(self.image.image_id, output_log['payload']['id'])