from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_separate_revert_optional_args(self):
    my_task = SeparateRevertOptionalTask()
    self.assertEqual(set(['execute_arg']), my_task.optional)
    self.assertEqual(set(['revert_arg']), my_task.revert_optional)