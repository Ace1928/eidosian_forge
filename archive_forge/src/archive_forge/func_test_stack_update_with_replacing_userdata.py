import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_with_replacing_userdata(self):
    """Test case for updating userdata of instance.

        Confirm that we can update userdata of instance during updating stack
        by the user of member role.

        Make sure that a resource that inherits from StackUser can be deleted
        during updating stack.
        """
    if not self.conf.minimal_image_ref:
        raise self.skipException('No minimal image configured to test')
    if not self.conf.minimal_instance_type:
        raise self.skipException('No flavor configured to test')
    parms = {'flavor': self.conf.minimal_instance_type, 'image': self.conf.minimal_image_ref, 'network': self.conf.fixed_network_name, 'user_data': ''}
    stack_identifier = self.stack_create(template=self.update_userdata_template, parameters=parms)
    parms_updated = parms
    parms_updated['user_data'] = 'two'
    self.update_stack(stack_identifier, template=self.update_userdata_template, parameters=parms_updated)