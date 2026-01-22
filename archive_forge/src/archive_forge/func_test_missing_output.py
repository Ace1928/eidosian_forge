import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
def test_missing_output(self):
    templ_missing_output = '\nheat_template_version: 2015-04-30\nparameters:\n  one:\n    type: string\n  two:\n    type: string\nresources:\n  str:\n    type: OS::Heat::RandomString\n'
    try:
        self.stack_create(template=self.template, environment=self.env, files={'facade.yaml': self.templ_facade, 'concrete.yaml': templ_missing_output}, expected_status='CREATE_FAILED')
    except heat_exceptions.HTTPBadRequest as exc:
        exp = 'ERROR: Attribute here-it-is for facade OS::Thingy missing in provider'
        self.assertEqual(exp, str(exc))