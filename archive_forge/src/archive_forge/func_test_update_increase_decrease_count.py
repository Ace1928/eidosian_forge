import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def test_update_increase_decrease_count(self):
    env = {'resource_registry': {'My::RandomString': 'OS::Heat::RandomString'}}
    create_template = self.template.replace('count: 0', 'count: 2')
    stack_identifier = self.stack_create(template=create_template, environment=env)
    self.assertEqual({u'random_group': u'OS::Heat::ResourceGroup'}, self.list_resources(stack_identifier))
    self._validate_resources(stack_identifier, 2)
    update_template = self.template.replace('count: 0', 'count: 5')
    self.update_stack(stack_identifier, update_template, environment=env)
    self._validate_resources(stack_identifier, 5)
    update_template = self.template.replace('count: 0', 'count: 3')
    self.update_stack(stack_identifier, update_template, environment=env)
    self._validate_resources(stack_identifier, 3)