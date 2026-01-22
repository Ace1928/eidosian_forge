import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_flavor_volume(self):
    parms = {'flavor': self.conf.minimal_instance_type, 'volume_size': 10, 'image': self.conf.minimal_image_ref, 'network': self.conf.fixed_network_name}
    stack_identifier = self.stack_create(template=test_template_updatae_flavor_and_volume_size, parameters=parms)
    parms_updated = parms
    parms_updated['volume_size'] = 20
    parms_updated['flavor'] = self.conf.instance_type
    self.update_stack(stack_identifier, template=test_template_updatae_flavor_and_volume_size, parameters=parms_updated)