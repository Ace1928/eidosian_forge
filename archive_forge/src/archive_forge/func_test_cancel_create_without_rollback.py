import copy
import eventlet
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_cancel_create_without_rollback(self):
    before, after = get_templates(delay_s=30)
    stack_id = self.stack_create(template=before, expected_status='CREATE_IN_PROGRESS')
    self._wait_for_resource_status(stack_id, 'test1', 'CREATE_IN_PROGRESS')
    self.cancel_update_stack(stack_id, rollback=False, expected_status='CREATE_FAILED')
    self.assertTrue(test.call_until_true(60, 2, self.verify_resource_status, stack_id, 'test1', 'CREATE_COMPLETE'))
    eventlet.sleep(2)
    self.assertTrue(self.verify_resource_status(stack_id, 'test2', 'INIT_COMPLETE'))