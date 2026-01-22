import copy
import json
import time
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
@test.requires_convergence
def test_retrigger_timeout(self):
    before, after = get_templates(delay_s=70)
    stack_id = self.stack_create(template=before, expected_status='CREATE_IN_PROGRESS', timeout=1)
    time.sleep(50)
    self.update_stack(stack_id, after)