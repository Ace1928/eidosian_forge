import copy
import json
import time
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
@test.requires_convergence
def test_nested_preemption(self):
    root_tmpl = _tmpl_with_rsrcs(preempt_root_rsrcs, preempt_root_out)
    files = {preempt_nested_stack_type: _tmpl_with_rsrcs(preempt_nested_rsrcs, preempt_nested_out), preempt_delay_stack_type: _tmpl_with_rsrcs(preempt_delay_rsrcs)}
    stack_id = self.stack_create(template=root_tmpl, files=files, parameters={input_param: 'foo'})
    delay_stack_uuid = self.get_stack_output(stack_id, 'delay_stack')
    self.update_stack(stack_id, template=root_tmpl, files=files, parameters={input_param: 'bar'}, expected_status='UPDATE_IN_PROGRESS')
    self._wait_for_resource_status(delay_stack_uuid, 'delay_resource', 'UPDATE_IN_PROGRESS')
    empty_nest_files = {preempt_nested_stack_type: _tmpl_with_rsrcs({})}
    self.update_stack(stack_id, template=root_tmpl, files=empty_nest_files, parameters={input_param: 'baz'})