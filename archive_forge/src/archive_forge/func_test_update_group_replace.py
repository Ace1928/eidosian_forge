import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
def test_update_group_replace(self):
    """Test case for ensuring non-updatable props case a replacement.

        Make sure that during a group update the non-updatable properties cause
        a replacement.
        """
    files = {'provider.yaml': self.instance_template}
    env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 1, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
    stack_identifier = self.stack_create(template=self.template, files=files, environment=env)
    rsrc = self.client.resources.get(stack_identifier, 'JobServerGroup')
    orig_asg_id = rsrc.physical_resource_id
    env2 = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': '2', 'AZ': 'wibble', 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type, 'user_data': 'new data'}}
    self.update_stack(stack_identifier, self.template, environment=env2, files=files)
    rsrc = self.client.resources.get(stack_identifier, 'JobServerGroup')
    self.assertNotEqual(orig_asg_id, rsrc.physical_resource_id)