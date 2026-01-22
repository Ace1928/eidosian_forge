import json
import os
from tempest.lib import exceptions
import yaml
from heatclient.tests.functional import base
def test_heat_resource_template_fmt_arg_long_json(self):
    ret = self.heat('resource-template --format json OS::Nova::Server')
    self.assertIn('"Type": "OS::Nova::Server"', ret)
    self.assertIsInstance(json.loads(ret), dict)