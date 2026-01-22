import copy
import json
from heatclient import exc
from oslo_log import log as logging
from testtools import matchers
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_group_suspend_resume(self):
    files = {'provider.yaml': self.instance_template}
    env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 4, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
    stack_identifier = self.stack_create(template=self.template, files=files, environment=env)
    nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
    self.stack_suspend(stack_identifier)
    self._wait_for_all_resource_status(nested_ident, 'SUSPEND_COMPLETE')
    self.stack_resume(stack_identifier)
    self._wait_for_all_resource_status(nested_ident, 'RESUME_COMPLETE')