from unittest import mock
from testtools import matchers
from oslo_service import periodic_task
from oslo_service.tests import base
@mock.patch('oslo_service.periodic_task.now')
def test_periodic_tasks_immediate_runs_now(self, mock_now):
    fake_time = 32503680000.0
    mock_now.return_value = fake_time

    class Manager(periodic_task.PeriodicTasks):

        @periodic_task.periodic_task(spacing=10, run_immediately=True)
        def bar(self, context):
            return 'bar'
    m = Manager(self.conf)
    self.assertEqual(1, len(m._periodic_tasks))
    task_name, task = m._periodic_tasks[0]
    self.assertEqual('bar', task_name)
    self.assertEqual(10, task._periodic_spacing)
    self.assertTrue(task._periodic_enabled)
    self.assertFalse(task._periodic_external_ok)
    self.assertTrue(task._periodic_immediate)
    self.assertIsNone(task._periodic_last_run)
    self.assertEqual(10, m._periodic_spacing[task_name])
    self.assertIsNone(m._periodic_last_run[task_name])
    idle = m.run_periodic_tasks(None)
    self.assertAlmostEqual(32503680000.0, m._periodic_last_run[task_name])
    self.assertAlmostEqual(10, idle, 1)
    mock_now.return_value = fake_time + 5
    idle = m.run_periodic_tasks(None)
    self.assertAlmostEqual(5, idle, 1)