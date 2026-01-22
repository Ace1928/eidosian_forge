import boto.swf.layer2
from boto.swf.layer2 import Decider, ActivityWorker
from tests.unit import unittest
from mock import Mock
def test_worker_override_tasklist(self):
    self.worker.poll(task_list='some_other_tasklist')
    self.worker._swf.poll_for_activity_task.assert_called_with('test', 'some_other_tasklist')