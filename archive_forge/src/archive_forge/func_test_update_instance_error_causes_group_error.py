import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
def test_update_instance_error_causes_group_error(self):
    """Test update failing a resource in the instance group.

        If a resource in an instance group fails to be created during an
        update, the instance group itself will fail and the broken inner
        resource will remain.
        """
    files = {'provider.yaml': self.instance_template}
    env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 2, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
    stack_identifier = self.stack_create(template=self.template, files=files, environment=env)
    initial_resources = {'JobServerConfig': 'AWS::AutoScaling::LaunchConfiguration', 'JobServerGroup': 'OS::Heat::InstanceGroup'}
    self.assertEqual(initial_resources, self.list_resources(stack_identifier))
    stack = self.client.stacks.get(stack_identifier)
    self.assert_instance_count(stack, 2)
    nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
    self._assert_instance_state(nested_ident, 2, 0)
    initial_list = [res.resource_name for res in self.client.resources.list(nested_ident)]
    env['parameters']['size'] = 3
    files2 = {'provider.yaml': self.bad_instance_template}
    self.client.stacks.update(stack_id=stack_identifier, template=self.template, files=files2, disable_rollback=True, parameters={}, environment=env)
    self._wait_for_stack_status(stack_identifier, 'UPDATE_FAILED')
    nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
    for res in self.client.resources.list(nested_ident):
        if res.resource_name in initial_list:
            self._wait_for_resource_status(nested_ident, res.resource_name, 'UPDATE_FAILED')
        else:
            self._wait_for_resource_status(nested_ident, res.resource_name, 'CREATE_FAILED')