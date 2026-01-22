import json
import os
from tempest.lib import exceptions
import yaml
from heatclient.tests.functional import base
def test_heat_resource_template_fmt_arg_short_yaml(self):
    ret = self.heat('resource-template -F yaml OS::Nova::Server')
    self.assertIn('Type: OS::Nova::Server', ret)
    self.assertIsInstance(yaml.safe_load(ret), dict)