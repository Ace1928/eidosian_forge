from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
@mock.patch.object(task.LOG, 'warning')
def test_update_progress_lower_bound(self, mocked_warning):
    result = []

    def progress_callback(event_type, details):
        result.append(details.pop('progress'))
    a_task = ProgressTask()
    a_task.notifier.register(task.EVENT_UPDATE_PROGRESS, progress_callback)
    a_task.execute([-1.0, -0.5, 0.0])
    self.assertEqual([0.0, 0.0, 0.0], result)
    self.assertEqual(2, mocked_warning.call_count)