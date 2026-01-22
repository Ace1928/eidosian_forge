import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
def test_create_instance_error_causes_group_error(self):
    """Test create failing a resource in the instance group.

        If a resource in an instance group fails to be created, the instance
        group itself will fail and the broken inner resource will remain.
        """
    stack_name = self._stack_rand_name()
    files = {'provider.yaml': self.bad_instance_template}
    env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 2, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
    self.client.stacks.create(stack_name=stack_name, template=self.template, files=files, disable_rollback=True, parameters={}, environment=env)
    self.addCleanup(self._stack_delete, stack_name)
    stack = self.client.stacks.get(stack_name)
    stack_identifier = '%s/%s' % (stack_name, stack.id)
    self._wait_for_stack_status(stack_identifier, 'CREATE_FAILED')
    initial_resources = {'JobServerConfig': 'AWS::AutoScaling::LaunchConfiguration', 'JobServerGroup': 'OS::Heat::InstanceGroup'}
    self.assertEqual(initial_resources, self.list_resources(stack_identifier))
    nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
    self._assert_instance_state(nested_ident, 0, 2)