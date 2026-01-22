from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def test_build_schedule(self):
    cron_trigger = mock.Mock()
    wf_input = {'name': 'foo', 'instance': 'myinst', 'parent_id': None}
    sched = self.backups._build_schedule(cron_trigger, wf_input)
    self.assertEqual(cron_trigger.name, sched.id)
    self.assertEqual(wf_input['name'], sched.name)
    self.assertEqual(wf_input['instance'], sched.instance)
    self.assertEqual(cron_trigger.workflow_input, sched.input)