from unittest import mock
import futurist
import glance_store
from oslo_config import cfg
from taskflow import engines
import glance.async_
from glance.async_ import taskflow_executor
from glance.common.scripts.image_import import main as image_import
from glance import domain
import glance.tests.utils as test_utils
def test_task_fail_upload(self):
    with mock.patch.object(image_import, 'set_image_data') as import_mock:
        import_mock.side_effect = IOError
        self.task_repo.get.return_value = self.task
        self.executor.begin_processing(self.task.task_id)
    self.assertEqual('failure', self.task.status)
    self.task_repo.save.assert_called_with(self.task)
    self.assertEqual(1, import_mock.call_count)