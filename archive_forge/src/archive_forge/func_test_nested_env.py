import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
def test_nested_env(self):
    main_templ = '\nheat_template_version: 2013-05-23\nresources:\n  secret1:\n    type: My::NestedSecret\noutputs:\n  secret-out:\n    value: { get_attr: [secret1, value] }\n'
    nested_templ = '\nheat_template_version: 2013-05-23\nresources:\n  secret2:\n    type: My::Secret\noutputs:\n  value:\n    value: { get_attr: [secret2, value] }\n'
    env_templ = '\nresource_registry:\n  "My::Secret": "OS::Heat::RandomString"\n  "My::NestedSecret": nested.yaml\n'
    stack_identifier = self.stack_create(template=main_templ, files={'nested.yaml': nested_templ}, environment=env_templ)
    nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'secret1')
    sec2 = self.client.resources.get(nested_ident, 'secret2')
    self.assertEqual('secret1', sec2.parent_resource)