from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_rebind_unknown_kwargs(self):
    my_task = KwargsTask(rebind={'foo': 'bar'})
    expected = {'foo': 'bar', 'spam': 'spam'}
    self.assertEqual(expected, my_task.rebind)