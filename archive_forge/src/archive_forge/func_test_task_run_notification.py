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
def test_task_run_notification(self):
    with mock.patch('glance.async_.TaskExecutor') as mock_executor:
        executor = mock_executor.return_value
        executor._run.return_value = mock.Mock()
        self.task_proxy.run(executor=mock_executor)
    output_logs = self.notifier.get_logs()
    self.assertEqual(1, len(output_logs))
    output_log = output_logs[0]
    self.assertEqual('INFO', output_log['notification_type'])
    self.assertEqual('task.run', output_log['event_type'])
    self.assertEqual(self.task.task_id, output_log['payload']['id'])
    self.assertNotIn(self.task.image_id, output_log['payload'])
    self.assertNotIn(self.task.user_id, output_log['payload'])
    self.assertNotIn(self.task.request_id, output_log['payload'])