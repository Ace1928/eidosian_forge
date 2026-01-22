import yaml
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack.tests.functional import base
from openstack.tests.functional.network.v2 import test_network
def test_suspend_resume(self):
    suspend_status = 'SUSPEND_COMPLETE'
    resume_status = 'RESUME_COMPLETE'
    self.conn.orchestration.suspend_stack(self.stack)
    sot = self.conn.orchestration.wait_for_status(self.stack, suspend_status, wait=self._wait_for_timeout)
    self.assertEqual(suspend_status, sot.status)
    self.conn.orchestration.resume_stack(self.stack)
    sot = self.conn.orchestration.wait_for_status(self.stack, resume_status, wait=self._wait_for_timeout)
    self.assertEqual(resume_status, sot.status)