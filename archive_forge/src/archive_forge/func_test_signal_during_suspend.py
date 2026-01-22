import copy
import json
from heatclient import exc
from oslo_log import log as logging
from testtools import matchers
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_signal_during_suspend(self):
    """Prove that a signal will fail when the stack is in suspend."""
    stack_identifier = self.stack_create(template=self.template, files=self.files, environment=self.env)
    self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 2))
    nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
    self.client.actions.suspend(stack_id=stack_identifier)
    self._wait_for_stack_status(stack_identifier, 'SUSPEND_COMPLETE')
    ex = self.assertRaises(exc.BadRequest, self.client.resources.signal, stack_identifier, 'ScaleUpPolicy')
    error_msg = 'Signal resource during SUSPEND is not supported'
    self.assertIn(error_msg, str(ex))
    ev = self.wait_for_event_with_reason(stack_identifier, reason='Cannot signal resource during SUSPEND', rsrc_name='ScaleUpPolicy')
    self.assertEqual('SUSPEND_COMPLETE', ev[0].resource_status)
    self._wait_for_stack_status(nested_ident, 'SUSPEND_COMPLETE')
    self._wait_for_stack_status(stack_identifier, 'SUSPEND_COMPLETE')
    self.assertTrue(test.call_until_true(self.build_timeout, self.build_interval, self.check_instance_count, stack_identifier, 2))