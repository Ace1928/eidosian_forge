import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_alias_type(self):
    env = {'resource_registry': {'My::TestResource': 'OS::Heat::RandomString', 'My::TestResource2': 'OS::Heat::RandomString'}}
    stack_identifier = self.stack_create(template=self.provider_template, environment=env)
    p_res = self.client.resources.get(stack_identifier, 'test1')
    self.assertEqual('My::TestResource', p_res.resource_type)
    initial_resources = {'test1': 'My::TestResource'}
    self.assertEqual(initial_resources, self.list_resources(stack_identifier))
    res = self.client.resources.get(stack_identifier, 'test1')
    tmpl_update = copy.deepcopy(self.provider_template)
    tmpl_update['resources']['test1']['type'] = 'My::TestResource2'
    self.update_stack(stack_identifier, tmpl_update, environment=env)
    res_a = self.client.resources.get(stack_identifier, 'test1')
    self.assertEqual(res.physical_resource_id, res_a.physical_resource_id)
    self.assertEqual(res.attributes['value'], res_a.attributes['value'])